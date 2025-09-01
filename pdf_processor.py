import os
import asyncio
import fitz
import argparse
from glob import glob

# Import the refactored converter classes
from text_based_converter_refactored import TextBasedConverter
from image_based_converter_refactored import ImageBasedConverter
from utils import normalize_NFC

class PdfProcessor:
    """
    Acts as the main controller to determine the PDF type and delegate the 
    conversion task to the appropriate specialized converter class.
    """
    def __init__(self, output_dir="output", gemini_api_key=None):
        self.output_dir = output_dir
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize converters with consistent interface
        self.text_converter = TextBasedConverter(output_dir=self.output_dir)
        self.image_converter = ImageBasedConverter(
            output_dir=self.output_dir, 
            api_key=self.gemini_api_key
        )

    def is_text_based(self, pdf_path: str, text_char_threshold: int = 100) -> bool:
        """
        Determines if a PDF is text-based by checking the average number of 
        characters on the first few pages. If the average is below the threshold,
        it's considered image-based.
        """
        try:
            doc = fitz.open(pdf_path)
            if doc.page_count == 0:
                return False
            
            total_chars = 0
            num_pages_to_check = min(len(doc), 3)

            for i in range(num_pages_to_check):
                page = doc.load_page(i)
                total_chars += len(page.get_text())
            
            doc.close()
            
            avg_chars_per_page = total_chars / num_pages_to_check
            print(f"[{os.path.basename(pdf_path)}] Average chars per page: {avg_chars_per_page:.2f}")
            
            return avg_chars_per_page > text_char_threshold
        except Exception as e:
            print(f"Error checking PDF text extractability for {os.path.basename(pdf_path)}: {e}. Treating as image-based.")
            return False

    async def process_pdf(self, pdf_path: str):
        """
        Processes a single PDF file by determining its type and then calling the
        appropriate converter's main `convert` method.
        """
        pdf_path = normalize_NFC(pdf_path)

        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found at {pdf_path}")
            return

        print(f"Processing PDF: {os.path.basename(pdf_path)}")
        if self.is_text_based(pdf_path):
            print("-> PDF is text-based. Using TextBasedConverter.")
            # This is a synchronous method and does not need `await`
            self.text_converter.convert(pdf_path)
        else:
            print("-> PDF is image-based. Using ImageBasedConverter.")
            # This is an asynchronous method
            await self.image_converter.convert(pdf_path)
        print(f"Finished processing {os.path.basename(pdf_path)}\n")

async def main(input_path: str, output_dir: str, gemini_api_key: str = None):
    """
    Main entry point for the script. Handles both single file and directory inputs.
    It creates a PdfProcessor instance and processes the files concurrently.
    """
    processor = PdfProcessor(output_dir=output_dir, gemini_api_key=gemini_api_key)

    if os.path.isdir(input_path):
        print(f"Processing all PDF files in directory: {input_path}")
        pdf_files = glob(os.path.join(input_path, "*.pdf"))
        if not pdf_files:
            print("No PDF files found in the directory.")
            return
        # Create a list of tasks to run concurrently
        tasks = [processor.process_pdf(pdf_file) for pdf_file in pdf_files]
        await asyncio.gather(*tasks)
    elif os.path.isfile(input_path) and input_path.lower().endswith('.pdf'):
        await processor.process_pdf(input_path)
    else:
        print(f"Error: The path '{input_path}' is not a valid PDF file or directory.")

if __name__ == '__main__':
    if os.environ.get("environment") == 'production':
        parser = argparse.ArgumentParser(
            description="Process a single PDF or a directory of PDFs, automatically determining whether to use a text-based or image-based conversion method."
        )
        parser.add_argument(
            "input_path", 
            help="Path to the input PDF file or a directory containing PDF files."
        )
        parser.add_argument(
            "--output_dir", 
            default="output", 
            help="Directory to save the output files (e.g., markdown, json, images). Defaults to 'output'."
        )
        parser.add_argument(
            "--gemini_api_key",
            help="Gemini API key for image-based PDF processing. If not provided, will use GEMINI_API_KEY environment variable."
        )
        args = parser.parse_args()

        # Enhanced error checking for API key
        gemini_api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            print("Warning: No Gemini API key provided. Image-based PDF conversion will fail.")
            print("Please provide --gemini_api_key argument or set GEMINI_API_KEY environment variable.")

        # Run the main asynchronous function
        asyncio.run(main(args.input_path, args.output_dir, gemini_api_key))
    else:
        pdf_path = "/Users/yoonhae/greenlabs/data-labs/experiments/engineering/pipelines/pdf-to-opensearch/chunk_output/chunk_배추_사진.pdf"
        pdf_path = "/Users/yoonhae/greenlabs/data-labs/experiments/engineering/pipelines/pdf-to-opensearch/pdfs/농업기술길잡이_양파_107_115.pdf"
        output_dir = "/Users/yoonhae/greenlabs/data-labs/experiments/engineering/pipelines/pdf-to-opensearch/output_v2"
        
        gemini_api_key = os.environ.get("GEMINI_API_KEY")

        dir_path = "/Users/yoonhae/greenlabs/data-labs/experiments/engineering/pipelines/pdf-to-opensearch/pdfs/"
        pdf_files = sorted(glob(os.path.join(dir_path, "*.pdf")))
        pdf_files = pdf_files
        for pdf_file in pdf_files:
            input_path = pdf_file
            asyncio.run(main(input_path, output_dir, gemini_api_key))

        # pdf_path = "/Users/yoonhae/greenlabs/data-labs/experiments/engineering/pipelines/pdf-to-opensearch/pdfs/농업기술길잡이_감.pdf"
        # input_path = pdf_path
        # asyncio.run(main(input_path, output_dir, gemini_api_key))
        
        # input_path = pdf_path
        # asyncio.run(main(input_path, output_dir, gemini_api_key))
        # image_converter = ImageBasedConverter(
        #     output_dir=output_dir, 
        #     api_key=gemini_api_key
        # )
        # image_converter.test(pdf_path)
