import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Iterator
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from opensearchpy import OpenSearch, RequestsHttpConnection, helpers
from opensearchpy.exceptions import OpenSearchException

# --- 로거 설정 ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# --- 유틸리티 함수 ---
def get_secretmanager(
    secret_name: str,
    region_name: str = "ap-northeast-2",
    deserialize_json: bool = False,
) -> str | dict[str, Any]:
    """
    AWS Secrets Manager에서 시크릿 값을 가져옵니다.

    Args:
        secret_name: 가져올 시크릿의 이름
        region_name: AWS 리전 이름
        deserialize_json: 시크릿 값을 JSON으로 파싱할지 여부

    Returns:
        시크릿 값 (문자열 또는 JSON 객체)

    Raises:
        ClientError: AWS API 호출 중 오류 발생 시
    """
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        secret_string = get_secret_value_response["SecretString"]
        if deserialize_json:
            return json.loads(secret_string)
        else:
            return secret_string
    except ClientError as e:
        logger.error(f"Secrets Manager에서 시크릿을 가져오는 데 실패했습니다: {e}")
        raise e


# --- 메인 클래스 ---
class OpenSearchLoader:
    """
    OpenSearch에 데이터를 효율적으로 생성, 적재, 삭제하는 클래스.

    이 클래스는 OpenSearch 인덱스 생성, 벡터 데이터 적재, 메타데이터 기반 문서 삭제 등의
    기능을 제공합니다. 임베딩 모델을 사용하여 텍스트를 벡터로 변환하고, 이를 OpenSearch에
    저장하는 기능을 포함합니다.

    Attributes:
        METADATA_FIELDS: 메타데이터 필드의 타입 정의 딕셔너리
        region: AWS 서비스 리전
        bedrock_region: Bedrock 서비스 리전
        os_host: OpenSearch 호스트 주소
        os_user: OpenSearch 사용자 이름
        os_password: OpenSearch 비밀번호
        os_port: OpenSearch 포트 번호
        os_client: OpenSearch 클라이언트
        bedrock_client: AWS Bedrock 클라이언트
        s3_client: AWS S3 클라이언트
    """

    METADATA_FIELDS: dict[str, dict[str, str]] = {
        "source_type": {"type": "keyword"},
        "source_uri": {"type": "text"},
        "source_title": {"type": "text"},
        "publication_date": {"type": "date"},
        "crop_name": {"type": "keyword"},
        "page_number": {"type": "integer"},
        "chunk_sequence": {"type": "integer"},
        "image_urls": {"type": "keyword"},  # 배열 형태로 저장
        "created_at": {"type": "date"},
    }

    def __init__(
        self,
        os_secret_name: str = "data/prod/airflow/variables/opensearch_fm-staging",
        region: str = "ap-northeast-2",
        bedrock_region: str = "us-west-2",
    ) -> None:
        """
        OpenSearchLoader를 초기화합니다.

        Args:
            os_secret_name: OpenSearch 인증 정보가 저장된 Secrets Manager의 시크릿 이름
            region: AWS 서비스(S3, Secrets Manager)를 위한 기본 리전
            bedrock_region: Bedrock Runtime 클라이언트를 위한 리전

        Raises:
            ValueError: OpenSearch 인증 정보를 가져오는 데 실패한 경우
            Exception: 클라이언트 초기화에 실패한 경우
        """
        self.region = region
        self.bedrock_region = bedrock_region

        try:
            os_account = get_secretmanager(
                os_secret_name, region_name=self.region, deserialize_json=True
            )
            if not isinstance(os_account, dict):
                raise ValueError("OpenSearch 인증 정보가 딕셔너리 형태가 아닙니다.")
            
            self.os_host = os_account["host"]
            self.os_user = os_account["id"]
            self.os_password = os_account["pw"]
            self.os_port = os_account["port"]
        except (ClientError, KeyError) as e:
            logger.error(f"OpenSearch 인증 정보 초기화 실패: {e}")
            raise ValueError("OpenSearch 인증 정보를 가져올 수 없습니다.") from e

        self.os_client = self._initialize_opensearch_client()
        self.bedrock_client = self._initialize_bedrock_client()
        self.s3_client = boto3.client("s3", region_name=self.region)
        self.embedding_info_cache: dict[str, tuple[str, int]] = {}

    def _initialize_opensearch_client(self) -> OpenSearch:
        """
        OpenSearch 클라이언트를 초기화하고 반환합니다.

        Returns:
            초기화된 OpenSearch 클라이언트

        Raises:
            Exception: 클라이언트 초기화에 실패한 경우
        """
        try:
            return OpenSearch(
                hosts=[{"host": self.os_host, "port": self.os_port}],
                http_auth=(self.os_user, self.os_password),
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                timeout=30,
                max_retries=3,
                retry_on_timeout=True,
            )
        except Exception as e:
            logger.error(f"OpenSearch 클라이언트 초기화 실패: {e}")
            raise

    def _initialize_bedrock_client(self) -> boto3.client:
        """
        AWS Bedrock Runtime 클라이언트를 초기화하고 반환합니다.

        Returns:
            초기화된 Bedrock Runtime 클라이언트

        Raises:
            Exception: 클라이언트 초기화에 실패한 경우
        """
        try:
            config = Config(retries={"max_attempts": 10, "mode": "standard"})
            return boto3.client(
                "bedrock-runtime", region_name=self.bedrock_region, config=config
            )
        except Exception as e:
            logger.error(f"Bedrock 클라이언트 초기화 실패: {e}")
            raise

    def create_index(
        self,
        name: str,
        embedding_dimension: int,
        embedding_model: str,
        description: str = "",
        vector_method: dict[str, Any] | None = None,
    ) -> None:
        """
        지정된 스키마에 따라 OpenSearch에 새로운 Index를 생성합니다.

        Args:
            name: 생성할 Index의 이름
            embedding_dimension: 벡터 필드의 차원 수
            embedding_model: Index 메타데이터에 기록될 임베딩 모델 이름
            description: Index 메타데이터에 기록될 설명
            vector_method: vector_embedding의 method 설정을 위한 딕셔너리
                           (e.g., {"name": "hnsw", "space_type": "l2", "engine": "faiss"})

        Raises:
            OpenSearchException: 인덱스 생성 중 오류 발생 시
        """
        if self.os_client.indices.exists(index=name):
            logger.info(f"Index '{name}' already exists. Skipping creation.")
            return

        # https://docs.opensearch.org/docs/latest/field-types/supported-field-types/knn-methods-engines/
        method_config = {
            "name": "hnsw",  # hnsw, ivf
            "space_type": "l2",  # l1, l2, linf, cosinesimil, innerproduct, hamming, hammingbit
            "engine": "faiss",  # faiss, lucene
        }
        if vector_method:
            method_config.update(vector_method)

        index_body = {
            "settings": {
                "index": {  # https://opendistro.github.io/for-elasticsearch-docs/docs/knn/settings/
                    "knn": True,
                    "knn.algo_param.ef_search": 512,  # default 512
                    "knn.algo_param.ef_construction": 512,  # default 512
                    "knn.algo_param.m": 16,  # default 16 (2 ~ 100)
                }
            },
            "mappings": {
                "_meta": {
                    "embedding_model": embedding_model,
                    "embedding_dimension": embedding_dimension,
                    "description": description,
                },
                "properties": {
                    "vector_embedding": {
                        "type": "knn_vector",
                        "dimension": embedding_dimension,
                        "method": method_config,
                    },
                    "chunk_text_current": {"type": "text"},
                    "chunk_text_previous": {"type": "text"},
                    "chunk_text_next": {"type": "text"},
                    "metadata": {"properties": self.METADATA_FIELDS},
                },
            },
        }
        try:
            self.os_client.indices.create(index=name, body=index_body)
            logger.info(f"Index '{name}' created successfully.")
        except OpenSearchException as e:
            logger.error(f"Failed to create index '{name}': {e}")
            raise

    
    def _get_embedding_info(self, index_name: str) -> tuple[str, int]:
        """
        인덱스 메타데이터에서 embedding 정보를 가져옵니다.
        """
        if index_name in self.embedding_info_cache:
            return self.embedding_info_cache[index_name]
        
        index_mapping = self.os_client.indices.get_mapping(index=index_name)
        index_meta = index_mapping.get(index_name, {}).get("mappings", {}).get("_meta", {})
        embedding_model_id = index_meta.get("embedding_model")
        embedding_dimension = index_meta.get("embedding_dimension")
        self.embedding_info_cache[index_name] = (embedding_model_id, embedding_dimension)
        return embedding_model_id, embedding_dimension
    

    def insert_chunk_list(
            self,
            index_name: str,
            chunk_list: list[str],
            base_metadata: dict[str, Any],
            embedding_model_id: str | None = None,
            embedding_dimension: int | None = None,
            local_save_path: str | None = None,
            s3_save_path: str | None = None,
            batch_size: int = 100,
    ) -> int:
        """
        청크 리스트를 임베딩하여 OpenSearch에 대량으로 적재(Insert)합니다.

        Args:
            index_name: 데이터를 적재할 Index 이름
            chunk_list: 적재할 청크 리스트
            base_metadata: 청크 리스트의 메타데이터
            embedding_model_id: 사용할 Bedrock 임베딩 모델 ID. None이면 인덱스 메타데이터에서 가져옴
            embedding_dimension: 임베딩 벡터의 차원 수. None이면 인덱스 메타데이터에서 가져옴
            local_save_path: 원본 데이터를 저장할 로컬 경로. "default" 사용 가능
            s3_save_path: 원본 데이터를 저장할 S3 경로. "default" 사용 가능
            batch_size: Bulk API 요청 당 문서 수

        Returns:
            성공적으로 인덱싱된 문서 수

        Raises:
            ValueError: 데이터 유효성 검사 실패 시
            OpenSearchException: 벌크 인덱싱 중 오류 발생 시
        """

        document_list = []
        current_chunk_num = 0

        prev_document = None
        current_document = None
        
        for chunk in chunk_list:
            sub_chunk_list = self._split_text(chunk)
            for sub_chunk in sub_chunk_list:
                current_chunk_num += 1
                sub_metadata = base_metadata.copy()
                sub_metadata["chunk_sequence"] = current_chunk_num
                current_document = {
                    "chunk_text_current": sub_chunk,
                    "chunk_text_previous": prev_document["chunk_text_current"] if prev_document is not None else None,
                    "metadata": sub_metadata,
                }
                if prev_document is not None:
                    prev_document["chunk_text_next"] = current_document["chunk_text_current"]

                document_list.append(current_document)
                prev_document = current_document
        
        return self.insert_document_list(
            index_name,
            document_list,
            embedding_model_id,
            embedding_dimension,
            local_save_path,
            s3_save_path,
            batch_size
        )


    def insert_document_list(
        self,
        index_name: str,
        data: list[dict[str, Any]],
        embedding_model_id: str | None = None,
        embedding_dimension: int | None = None,
        local_save_path: str | None = None,
        s3_save_path: str | None = None,
        batch_size: int = 100,
    ) -> int:
        """
        데이터를 임베딩하여 OpenSearch에 대량으로 적재(Insert)합니다.

        Args:
            index_name: 데이터를 적재할 Index 이름
            data: 적재할 데이터 리스트
            embedding_model_id: 사용할 Bedrock 임베딩 모델 ID. None이면 인덱스 메타데이터에서 가져옴
            embedding_dimension: 임베딩 벡터의 차원 수. None이면 인덱스 메타데이터에서 가져옴
            local_save_path: 원본 데이터를 저장할 로컬 경로. "default" 사용 가능
            s3_save_path: 원본 데이터를 저장할 S3 경로. "default" 사용 가능
            batch_size: Bulk API 요청 당 문서 수

        Returns:
            성공적으로 인덱싱된 문서 수

        Raises:
            ValueError: 데이터 유효성 검사 실패 시
            OpenSearchException: 벌크 인덱싱 중 오류 발생 시
        """
        self._validate_data(data)

        # 날짜 기반 폴더와 시간 정보가 포함된 파일명
        date_folder = datetime.now(timezone.utc).strftime("%Y%m%d")  # 날짜만 (YYYYMMDD)
        time_suffix = datetime.now(timezone.utc).strftime("%H%M%S")  # 시간 (HHMMSS)

        if local_save_path:
            self._save_to_local(
                data, index_name, local_save_path, date_folder, time_suffix
            )
        if s3_save_path:
            self._save_to_s3(data, index_name, s3_save_path, date_folder, time_suffix)

        if embedding_model_id is None or embedding_dimension is None:
            embedding_model_id, embedding_dimension = self._get_embedding_info(index_name)

        actions = self._generate_bulk_actions(
            data, index_name, embedding_model_id, embedding_dimension
        )

        try:
            success, _ = helpers.bulk(
                self.os_client, actions, chunk_size=batch_size, raise_on_error=True
            )
            logger.info(f"Successfully indexed {success} documents.")
            return success
        except OpenSearchException as e:
            logger.error(f"Bulk indexing failed: {e}")
            raise e

    def build_term_query(self, field_name: str, value: Any) -> dict[str, Any]:
        """
        필드 타입에 따라 적절한 쿼리를 생성합니다.

        Args:
            field_name: 쿼리할 필드 이름
            value: 쿼리 값

        Returns:
            생성된 쿼리 딕셔너리
        """
        field_info = self.METADATA_FIELDS.get(field_name, {})
        field_type = field_info.get("type")

        if field_type == "keyword":
            return {"term": {f"metadata.{field_name}": value}}
        elif field_type == "text":
            if "fields" in field_info and "keyword" in field_info["fields"]:
                return {"term": {f"metadata.{field_name}.keyword": value}}
            else:
                return {"match_phrase": {f"metadata.{field_name}": value}}
        else:
            return {"term": {f"metadata.{field_name}": value}}

    def delete_documents_by_metadata(
        self, index_name: str, metadata_filter: dict[str, Any]
    ) -> int:
        """
        메타데이터 조건과 일치하는 문서를 Index에서 삭제합니다.

        Args:
            index_name: 문서를 삭제할 Index 이름
            metadata_filter: 삭제할 문서를 필터링하기 위한 조건

        Returns:
            삭제된 문서의 수

        Raises:
            ValueError: 필수 메타데이터 필드가 없는 경우
            OpenSearchException: 문서 삭제 중 오류 발생 시
        """
        if not all(k in metadata_filter for k in ["source_type", "source_uri"]):
            raise ValueError(
                "metadata_filter must contain 'source_type' and 'source_uri'."
            )

        query = {
            "query": {
                "bool": {
                    "filter": [
                        self.build_term_query(key, value)
                        for key, value in metadata_filter.items()
                    ]
                }
            }
        }
        try:
            response = self.os_client.delete_by_query(index=index_name, body=query)
            deleted_count = response.get("deleted", 0)
            logger.info(
                f"Deleted {deleted_count} documents from '{index_name}' matching filter."
            )
            return deleted_count
        except OpenSearchException as e:
            logger.error(f"Failed to delete documents by metadata: {e}")
            raise

    def _validate_data(self, data: list[dict[str, Any]]) -> None:
        """
        데이터 유효성을 검사합니다.

        Args:
            data: 검사할 데이터 리스트

        Raises:
            ValueError: 필수 필드가 없는 경우
        """
        required_fields = ["chunk_text_current", "metadata"]
        required_metadata_fields = ["source_type", "source_uri"]
        for i, item in enumerate(data):
            for field in required_fields:
                if field not in item:
                    raise ValueError(
                        f"Missing required field '{field}' in data item {i}."
                    )

            metadata = item["metadata"]
            for field in required_metadata_fields:
                if field not in metadata:
                    raise ValueError(
                        f"Missing required metadata field '{field}' in data item {i}."
                    )

            if metadata.get("created_at") is None:
                metadata["created_at"] = datetime.now(timezone.utc).isoformat()

    def _split_text(self, text: str, chunk_size: int = 2000) -> list[str]:
        # 1000자 길이 제한을 고려하여 텍스트를 문장 단위로 분할합니다
        # 마침표(.)와 줄바꿈(\n)을 기준으로 분할
        sentences = re.split(r'(?<=\.)\s+|\n+', text)
        chunks = []
        current_text_list = []
        current_length = 0
        
        for sentence in sentences:
            # 빈 문장이나 공백만 있는 문장은 건너뛰기
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if current_length + len(sentence) <= chunk_size:
                current_text_list.append(sentence)
                current_length += len(sentence)
            else:
                if current_text_list:
                    chunks.append(" ".join(current_text_list))

                # 한 문장 자체가 max length 보다 길면 그냥 split
                if len(sentence) > chunk_size:
                    chunks.extend([sentence[i : i + chunk_size] for i in range(0, len(sentence), chunk_size)])
                    current_text_list = []
                    current_length = 0
                else:
                    current_text_list = [sentence]
                    current_length = len(sentence)
        
        if current_text_list:
            chunks.append(" ".join(current_text_list))
            
        if not chunks:
            chunks = [text]  # 분할할 수 없는 경우 원본 텍스트 사용
        
        return chunks

    def _get_embedding(
        self,
        text: str,
        model_id: str,
        embedding_dimension: int | None = None,
        input_type: str = "search_document",
    ) -> list[float]:
        """
        주어진 텍스트에 대한 임베딩 벡터를 반환합니다.

        Args:
            text: 임베딩할 텍스트
            model_id: 사용할 임베딩 모델 ID
            embedding_dimension: 임베딩 벡터의 차원 수 (Titan 모델에 필요)
            input_type: 임베딩 입력 타입 (Cohere 모델에 필요)

        Returns:
            생성된 임베딩 벡터

        Raises:
            ValueError: 지원하지 않는 모델 ID인 경우
            ClientError: Bedrock API 호출 중 오류 발생 시
            KeyError: 응답에서 임베딩을 찾을 수 없는 경우
        """
        if "cohere" in model_id:
            text_list = self._split_text(text)
            body = json.dumps({"input_type": input_type, "texts": text_list})
        elif "titan" in model_id:
            body = json.dumps(
                {"inputText": text, "dimensions": embedding_dimension or 1024}
            )
        else:
            raise ValueError(f"Unsupported model ID: {model_id}")

        try:
            response = self.bedrock_client.invoke_model(
                body=body,
                modelId=model_id,
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response["body"].read())
            return response_body["embeddings"]
        except (ClientError, KeyError) as e:
            logger.error(f"Failed to get embedding for text '{text[:30]}...': {e}")
            raise

    def _generate_bulk_actions(
        self,
        data: list[dict[str, Any]],
        index_name: str,
        embedding_model_id: str,
        embedding_dimension: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Bulk API를 위한 actions 이터레이터를 생성합니다.

        Args:
            data: 인덱싱할 데이터 리스트
            index_name: 인덱싱할 인덱스 이름
            embedding_model_id: 사용할 임베딩 모델 ID
            embedding_dimension: 임베딩 벡터의 차원 수

        Yields:
            Bulk API에 사용할 액션 딕셔너리
        """
        for i, item in enumerate(data):
            logger.info(f"Processing item {i}: {item['metadata']['source_uri']}")
            embedding = self._get_embedding(
                item["chunk_text_current"], embedding_model_id, embedding_dimension
            )
            logger.info(f"Embedding generated: {len(embedding)} dimensions")
            item["vector_embedding"] = embedding[0]

            # # Generate a unique ID
            # doc_id_base = f"{item['metadata']['source_uri']}_{item['metadata'].get('chunk_sequence', 0)}_{hashlib.md5(item['chunk_text_current'].encode('utf-8')).hexdigest()}"
            # doc_id = doc_id_base.replace("/", "_").replace(":", "_")
            # logger.info(f"Generated document ID: {doc_id}")

            action = {
                "_op_type": "index",
                "_index": index_name,
                # "_id": doc_id,
                "_source": item,
            }
            logger.debug(f"Action: {action}")
            yield action

    def _save_to_local(
        self,
        data: list[dict[str, Any]],
        index_name: str,
        path: str,
        date_folder: str,
        time_suffix: str,
    ) -> None:
        """
        데이터를 로컬 파일 시스템에 저장합니다.

        Args:
            data: 저장할 데이터 리스트
            index_name: 인덱스 이름
            path: 저장할 경로 ("default"는 기본 경로 사용)
            date_folder: 날짜 폴더 이름 (YYYYMMDD 형식)
            time_suffix: 시간 접미사 (HHMMSS 형식)

        Raises:
            IOError: 파일 저장 중 오류 발생 시
        """
        source_type = data[0]["metadata"].get("source_type", "unknown")
        if path == "default":
            base_path = os.path.join(
                ".", "archives", index_name, source_type, date_folder, time_suffix
            )
        else:
            base_path = path

        os.makedirs(base_path, exist_ok=True)

        for item in data:
            uri = item["metadata"]["source_uri"]
            seq = item["metadata"].get("chunk_sequence", 0)
            filename = f"{os.path.basename(urlparse(uri).path)}_{seq}.json"
            filepath = os.path.join(base_path, filename)
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(item, f, ensure_ascii=False, indent=4)
            except IOError as e:
                logger.error(f"Failed to save data to local file {filepath}: {e}")

    def _save_to_s3(
        self,
        data: list[dict[str, Any]],
        index_name: str,
        path: str,
        date_folder: str,
        time_suffix: str,
    ) -> None:
        """
        데이터를 S3에 저장합니다.

        Args:
            data: 저장할 데이터 리스트
            index_name: 인덱스 이름
            path: 저장할 S3 경로 ("default"는 기본 버킷 사용)
            date_folder: 날짜 폴더 이름 (YYYYMMDD 형식)
            time_suffix: 시간 접미사 (HHMMSS 형식)

        Raises:
            ClientError: S3 API 호출 중 오류 발생 시
        """
        if path == "default":
            bucket_name = os.environ.get("DEFAULT_S3_BUCKET", "greenlabs-data-lake")
            if not bucket_name:
                logger.error(
                    "DEFAULT_S3_BUCKET environment variable not set. Cannot save to default S3 path."
                )
                return
            source_type = data[0]["metadata"].get("source_type", "unknown")
            base_path = (
                f"archives/{index_name}/{source_type}/{date_folder}/{time_suffix}"
            )
        else:
            parsed_s3_path = urlparse(path)
            bucket_name = parsed_s3_path.netloc
            base_path = parsed_s3_path.path.lstrip("/")

        for item in data:
            uri = item["metadata"]["source_uri"]
            seq = item["metadata"].get("chunk_sequence", 0)
            filename = f"{os.path.basename(urlparse(uri).path)}_{seq}.json"
            s3_key = f"{base_path}/{filename}"
            try:
                self.s3_client.put_object(
                    Bucket=bucket_name,
                    Key=s3_key,
                    Body=json.dumps(item, ensure_ascii=False, indent=4),
                    ContentType="application/json",
                )
            except ClientError as e:
                logger.error(
                    f"Failed to save data to S3 path s3://{bucket_name}/{s3_key}: {e}"
                )


def run_total_test():
    """
    OpenSearchLoader 클래스의 주요 기능을 테스트하는 함수.
    """
    # --- 테스트 설정 ---
    # !!! 보안 주의: 실제 프로덕션 시크릿 이름 사용에 유의하세요.
    # 테스트용 또는 개발용 OpenSearch 클러스터와 시크릿을 사용하는 것을 권장합니다.
    OS_SECRET_NAME = "prod/opensearch/data-admin"
    TEST_INDEX_NAME = f"test-index-{datetime.now().strftime('%Y%m%d')}"
    EMBEDDING_MODEL_ID = "cohere.embed-multilingual-v3"
    EMBEDDING_DIMENSION = 1024

    print("--- OpenSearchLoader Test Start ---")

    loader = None
    try:
        # --- 1. OpenSearchLoader 인스턴스 생성 ---
        print(
            f"\n[Step 1] Initializing OpenSearchLoader with secret: '{OS_SECRET_NAME}'..."
        )
        loader = OpenSearchLoader(os_secret_name=OS_SECRET_NAME)
        print("-> Initialized successfully.")

        # --- 2. Index 생성 테스트 ---
        print(f"\n[Step 2] Creating index '{TEST_INDEX_NAME}' with default settings...")
        loader.create_index(
            name=TEST_INDEX_NAME,
            embedding_dimension=EMBEDDING_DIMENSION,
            embedding_model=EMBEDDING_MODEL_ID,
            description="Test index created by automated test script.",
        )
        # 한 번 더 호출하여 이미 존재할 때 건너뛰는지 확인
        loader.create_index(
            name=TEST_INDEX_NAME,
            embedding_dimension=EMBEDDING_DIMENSION,
            embedding_model=EMBEDDING_MODEL_ID,
        )
        print(f"-> Index '{TEST_INDEX_NAME}' created or already exists.")

        # --- 3. 데이터 적재 테스트 ---
        print("\n[Step 3] Inserting sample data...")
        test_data = [
            {
                "chunk_text_current": "첫 번째 테스트 문서입니다. 딸기에 대한 내용입니다.",
                "chunk_text_previous": None,
                "chunk_text_next": "이것은 다음 내용입니다.",
                "metadata": {
                    "source_type": "WEB_ARTICLE",
                    "source_uri": "test://suite/doc/111",
                    "source_title": "테스트 문서 1",
                    "publication_date": "2024-07-15T00:00:00Z",
                    "crop_name": "딸기",
                    "chunk_sequence": 1,
                    "image_urls": [
                        "https://example.com/image1.jpg",
                        "https://example.com/image2.jpg",
                    ],
                },
            },
            {
                "chunk_text_current": "두 번째 테스트 문서입니다. 토마토에 대한 내용입니다.",
                "chunk_text_previous": "이것은 이전 내용입니다.",
                "chunk_text_next": None,
                "metadata": {
                    "source_type": "WEB_ARTICLE",
                    "source_uri": "test://suite/doc/222",
                    "source_title": "테스트 문서 2",
                    "publication_date": "2024-07-14T00:00:00Z",
                    "crop_name": "토마토",
                    "chunk_sequence": 1,
                },
            },
        ]
        loader.insert_document_list(
            index_name=TEST_INDEX_NAME,
            data=test_data,
            embedding_model_id=EMBEDDING_MODEL_ID,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        # OpenSearch가 색인을 처리할 시간을 줍니다.
        import time

        time.sleep(2)

        count_res = loader.os_client.count(index=TEST_INDEX_NAME)
        print(f"-> Inserted {count_res['count']} documents.")
        assert count_res["count"] == 2

        # --- 4. 메타데이터 기반 삭제 테스트 ---
        print("\n[Step 4] Deleting documents by metadata...")
        metadata_filter = {
            "source_type": "WEB_ARTICLE",
            "source_uri": "test://suite/doc/111",
            "source_title": "테스트 문서 1",
        }
        deleted_count = loader.delete_documents_by_metadata(
            index_name=TEST_INDEX_NAME, metadata_filter=metadata_filter
        )
        print(f"-> API reported {deleted_count} documents for deletion.")

        time.sleep(2)

        count_res = loader.os_client.count(index=TEST_INDEX_NAME)
        print(f"-> Remaining documents after deletion: {count_res['count']}.")

        print("\n--- Test Scenario Passed ---")

    except Exception as e:
        logger.error(f"An error occurred during the test: {e}", exc_info=True)
        print("\n--- Test Scenario Failed ---")

    finally:
        # # --- 5. 테스트 Index 삭제 (정리) ---
        # if loader and loader.os_client.indices.exists(index=TEST_INDEX_NAME):
        #     print(f"\n[Cleanup] Deleting test index '{TEST_INDEX_NAME}'...")
        #     loader.os_client.indices.delete(index=TEST_INDEX_NAME)
        #     print("-> Test index deleted.")

        print("\n--- OpenSearchLoader Test End ---")


def run_insert_vectors(index_name: str = "test-index-20250714"):
    """
    지정된 인덱스의 _meta 정보를 읽어와서 해당 설정으로 문서를 삽입하는 테스트 함수.

    Args:
        index_name: 테스트할 인덱스 이름. 없으면 사용자에게 입력받습니다.
    """
    # --- 테스트 설정 ---
    loader = OpenSearchLoader()

    test_data = [
        {
            "chunk_text_current": "이것은 _meta 정보를 활용한 테스트 문서입니다. 파프리카에 대한 내용입니다.",
            "chunk_text_previous": None,
            "chunk_text_next": "다음 내용은 파프리카 재배 방법에 대한 설명입니다.",
            "metadata": {
                "source_type": "WEB_ARTICLE",
                "source_uri": "test://meta_test/doc/333",
                "source_title": "메타 테스트 문서",
                "publication_date": datetime.now(timezone.utc).isoformat(),
                "crop_name": "파프리카",
                "chunk_sequence": 3,
                "image_urls": [
                    "https://example.com/paprika1.jpg",
                    "https://example.com/paprika2.jpg",
                ],
            },
        }
    ]

    success = loader.insert_document_list(
        index_name=index_name,
        data=test_data,
        local_save_path="default",
        s3_save_path="default",
    )
    print(f"-> Inserted {success} documents.")

    # OpenSearch가 색인을 처리할 시간을 줍니다.
    import time

    time.sleep(2)

    # --- 4. 삽입 확인 ---
    print("\n[Step 4] Verifying document insertion...")

    # 메타데이터 필터를 사용하여 방금 삽입한 문서 검색
    query = {
        "query": {
            "bool": {
                "filter": [
                    loader.build_term_query("source_uri", "test://meta_test/doc/333")
                ]
            }
        }
    }

    search_result = loader.os_client.search(index=index_name, body=query)
    hits = search_result.get("hits", {}).get("hits", [])

    if hits:
        print("-> Successfully inserted and retrieved document:")
        print(f"   - Document ID: {hits[0]['_id']}")
        print(f"   - Score: {hits[0]['_score']}")
        print(f"   - Source URI: {hits[0]['_source']['metadata']['source_uri']}")

    else:
        print("-> Error: Could not find the inserted document.")
    