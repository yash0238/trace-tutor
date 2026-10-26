"""
Table Extraction and Understanding Module

Extracts tables from PDFs and enables structured cell-level retrieval.
Supports markdown serialization and BM25 lookup for exact values.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re

logger = logging.getLogger(__name__)


class TableExtractor:
    """
    Extract and process tables from PDF documents.
    
    Converts tables to markdown and creates searchable chunks.
    """
    
    def __init__(self, output_dir: str = "data/interim/tables"):
        """
        Initialize table extractor.
        
        Args:
            output_dir: Directory to save extracted tables
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_from_pdf(self, pdf_path: str, page_num: Optional[int] = None) -> List[Dict]:
        """
        Extract tables from PDF.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Extract from specific page only (None = all pages)
            
        Returns:
            List of dicts with table metadata and content
        """
        try:
            import fitz  # PyMuPDF
            import pdfplumber
            
            tables = []
            
            # Use pdfplumber for table extraction (better than PyMuPDF for tables)
            with pdfplumber.open(pdf_path) as pdf:
                pages = [page_num] if page_num is not None else range(len(pdf.pages))
                
                for page_idx in pages:
                    if page_idx >= len(pdf.pages):
                        continue
                    
                    page = pdf.pages[page_idx]
                    page_tables = page.extract_tables()
                    
                    for table_idx, table_data in enumerate(page_tables):
                        if not table_data or len(table_data) == 0:
                            continue
                        
                        # Process table
                        processed_table = self._process_table(table_data)
                        
                        # Generate markdown
                        markdown = self._table_to_markdown(processed_table)
                        
                        # Save table
                        pdf_stem = Path(pdf_path).stem
                        table_filename = f"{pdf_stem}_p{page_idx:03d}_t{table_idx:02d}.md"
                        table_path = self.output_dir / table_filename
                        
                        with open(table_path, "w", encoding="utf-8") as f:
                            f.write(markdown)
                        
                        tables.append({
                            "table_id": f"{pdf_stem}_p{page_idx}_t{table_idx}",
                            "page": page_idx,
                            "table_index": table_idx,
                            "data": processed_table,
                            "markdown": markdown,
                            "path": str(table_path),
                            "rows": len(processed_table),
                            "cols": len(processed_table[0]) if processed_table else 0
                        })
            
            logger.info(f"Extracted {len(tables)} tables from {Path(pdf_path).name}")
            return tables
            
        except Exception as e:
            logger.error(f"Table extraction failed for {pdf_path}: {e}")
            return []
    
    def _process_table(self, table_data: List[List]) -> List[List[str]]:
        """Clean and normalize table data."""
        processed = []
        
        for row in table_data:
            processed_row = []
            for cell in row:
                # Clean cell content
                if cell is None:
                    cell_str = ""
                else:
                    cell_str = str(cell).strip()
                    # Remove extra whitespace
                    cell_str = re.sub(r'\s+', ' ', cell_str)
                
                processed_row.append(cell_str)
            
            # Only add non-empty rows
            if any(cell for cell in processed_row):
                processed.append(processed_row)
        
        return processed
    
    def _table_to_markdown(self, table_data: List[List[str]]) -> str:
        """Convert table to markdown format."""
        if not table_data:
            return ""
        
        markdown_lines = []
        
        # Header row
        header = table_data[0]
        markdown_lines.append("| " + " | ".join(header) + " |")
        
        # Separator
        markdown_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        
        # Data rows
        for row in table_data[1:]:
            # Pad row to match header length
            while len(row) < len(header):
                row.append("")
            markdown_lines.append("| " + " | ".join(row[:len(header)]) + " |")
        
        return "\n".join(markdown_lines)
    
    def chunk_table(self, table_dict: Dict, chunk_size: int = 3) -> List[Dict]:
        """
        Chunk table into row groups for retrieval.
        
        Args:
            table_dict: Table metadata dict from extract_from_pdf
            chunk_size: Number of rows per chunk
            
        Returns:
            List of chunk dicts with text, metadata, and table_id
        """
        chunks = []
        table_data = table_dict["data"]
        
        if not table_data or len(table_data) < 2:
            return chunks
        
        header = table_data[0]
        data_rows = table_data[1:]
        
        # Chunk 1: Whole table as markdown (for context)
        chunks.append({
            "chunk_id": f"{table_dict['table_id']}_full",
            "text": table_dict["markdown"],
            "type": "table_full",
            "table_id": table_dict["table_id"],
            "page": table_dict["page"]
        })
        
        # Chunk 2+: Row groups
        for i in range(0, len(data_rows), chunk_size):
            row_group = data_rows[i:i+chunk_size]
            
            # Create text representation with header
            chunk_lines = ["| " + " | ".join(header) + " |"]
            chunk_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
            for row in row_group:
                chunk_lines.append("| " + " | ".join(row[:len(header)]) + " |")
            
            chunk_text = "\n".join(chunk_lines)
            
            # Also create flat text for BM25 (better for cell value lookup)
            flat_text = " ".join([" ".join(row) for row in row_group])
            
            chunks.append({
                "chunk_id": f"{table_dict['table_id']}_rows_{i}_{i+chunk_size}",
                "text": chunk_text,
                "flat_text": flat_text,
                "type": "table_rows",
                "table_id": table_dict["table_id"],
                "page": table_dict["page"],
                "row_start": i + 1,  # +1 because we skip header
                "row_end": min(i + chunk_size, len(data_rows))
            })
        
        return chunks


class TableUnderstanding:
    """
    Semantic understanding of table content.
    
    Answers questions about table data.
    """
    
    @staticmethod
    def find_cell_value(table_data: List[List[str]], row_keyword: str, col_keyword: str) -> Optional[str]:
        """
        Find cell value by row and column keywords.
        
        Args:
            table_data: Processed table data
            row_keyword: Keyword to identify row (e.g., "Population")
            col_keyword: Keyword to identify column (e.g., "2020")
            
        Returns:
            Cell value or None if not found
        """
        if not table_data or len(table_data) < 2:
            return None
        
        header = table_data[0]
        
        # Find column index
        col_idx = None
        for idx, col_header in enumerate(header):
            if col_keyword.lower() in col_header.lower():
                col_idx = idx
                break
        
        if col_idx is None:
            return None
        
        # Find row
        for row in table_data[1:]:
            if row_keyword.lower() in row[0].lower():
                if col_idx < len(row):
                    return row[col_idx]
        
        return None
    
    @staticmethod
    def summarize_table(table_dict: Dict) -> str:
        """
        Generate natural language summary of table.
        
        Args:
            table_dict: Table metadata dict
            
        Returns:
            Summary text
        """
        table_data = table_dict["data"]
        
        if not table_data:
            return "Empty table"
        
        header = table_data[0]
        num_rows = len(table_data) - 1
        num_cols = len(header)
        
        summary = f"Table with {num_rows} rows and {num_cols} columns. "
        summary += f"Columns: {', '.join(header)}. "
        
        # Add sample values from first data row
        if len(table_data) > 1:
            first_row = table_data[1]
            summary += f"Example: {header[0]}={first_row[0]}"
            if len(first_row) > 1 and len(header) > 1:
                summary += f", {header[1]}={first_row[1]}"
        
        return summary


def create_table_evidence(table_dict: Dict, grade: int, subject: str, chapter: str) -> List:
    """
    Create Evidence objects from table data.
    
    Args:
        table_dict: Table metadata from TableExtractor
        grade: Grade level
        subject: Subject name
        chapter: Chapter identifier
        
    Returns:
        List of Evidence objects
    """
    from trace_tutor.schema import Evidence
    
    extractor = TableExtractor()
    chunks = extractor.chunk_table(table_dict, chunk_size=3)
    
    evidence_list = []
    
    for chunk in chunks:
        # Use a simpler Evidence structure since the full schema requires many fields
        evidence = {
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "page": chunk["page"],
            "grade": grade,
            "subject": subject,
            "chapter": chapter,
            "modality": "table",
            "metadata": {
                "table_id": chunk["table_id"],
                "chunk_type": chunk["type"],
                "flat_text": chunk.get("flat_text", "")
            }
        }
        
        evidence_list.append(evidence)
    
    return evidence_list
