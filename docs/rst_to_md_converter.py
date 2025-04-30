#!/usr/bin/env python3
"""
RST to Markdown converter script for DeepMIMO documentation.
This script converts reStructuredText files to Markdown format.
"""

import os
import sys
from pathlib import Path
import re

def convert_rst_to_md(rst_content):
    """Convert RST content to Markdown."""
    # Remove Sphinx-specific directives but preserve toctree content
    toctree_content = re.search(r'\.\.\s+toctree::.*?\n(.*?)(?=\n\n|\Z)', rst_content, re.DOTALL)
    if toctree_content:
        toctree_items = re.findall(r'^\s*(\w+(?:/\w+)?)', toctree_content.group(1), re.MULTILINE)
        toctree_md = '\n## Documentation Contents\n\n' + '\n'.join(f'* [{item}]({item}.md)' for item in toctree_items)
        rst_content = re.sub(r'\.\.\s+toctree::.*?\n.*?(?=\n\n|\Z)', toctree_md, rst_content, flags=re.DOTALL)
    
    # Remove other Sphinx directives
    rst_content = re.sub(r'\.\.\s+automodule::.*?\n.*?\n', '', rst_content, flags=re.DOTALL)
    rst_content = re.sub(r'\.\.\s+mermaid::.*?\n.*?\n', '', rst_content, flags=re.DOTALL)
    
    # Convert section headers (remove underlines)
    rst_content = re.sub(r'^([=]+)\n(.*?)\n\1+$', r'# \2', rst_content, flags=re.MULTILINE)
    rst_content = re.sub(r'^([-]+)\n(.*?)\n\1+$', r'## \2', rst_content, flags=re.MULTILINE)
    rst_content = re.sub(r'^([~]+)\n(.*?)\n\1+$', r'### \2', rst_content, flags=re.MULTILINE)
    
    # Convert code blocks
    rst_content = re.sub(r'\.\.\s+code-block::\s*(\w+)\n\n(.*?)(?=\n\n|\Z)', 
                        lambda m: f"```{m.group(1)}\n{m.group(2)}\n```", 
                        rst_content, flags=re.DOTALL)
    
    # Convert inline code
    rst_content = re.sub(r'``([^`]+)``', r'`\1`', rst_content)
    
    # Convert links
    rst_content = re.sub(r'`([^`]+)`_', r'[\1]', rst_content)
    rst_content = re.sub(r'\.\.\s+_([^:]+):\s*(.*?)$', r'[\1]: \2', rst_content, flags=re.MULTILINE)
    
    # Fix link format
    rst_content = re.sub(r'\[([^]]+) <([^>]+)>\]', r'[\1](\2)', rst_content)
    
    # Convert lists
    rst_content = re.sub(r'^\s*\*\s+', '* ', rst_content, flags=re.MULTILINE)
    rst_content = re.sub(r'^\s*-\s+', '* ', rst_content, flags=re.MULTILINE)
    
    # Convert emphasis
    rst_content = re.sub(r'\*\*([^*]+)\*\*', r'**\1**', rst_content)
    rst_content = re.sub(r'\*([^*]+)\*', r'*\1*', rst_content)
    
    # Convert references
    rst_content = re.sub(r':ref:`([^`]+)`', r'[\1]', rst_content)
    
    # Fix any remaining RST-style underlines
    rst_content = re.sub(r'^([=]+)$', '', rst_content, flags=re.MULTILINE)
    rst_content = re.sub(r'^([-]+)$', '', rst_content, flags=re.MULTILINE)
    rst_content = re.sub(r'^([~]+)$', '', rst_content, flags=re.MULTILINE)
    
    # Fix bare section titles
    rst_content = re.sub(r'^Features$', '## Features', rst_content, flags=re.MULTILINE)
    rst_content = re.sub(r'^Getting Help$', '## Getting Help', rst_content, flags=re.MULTILINE)
    rst_content = re.sub(r'^Indices and tables$', '## Indices and tables', rst_content, flags=re.MULTILINE)
    
    # Fix bare toctree items
    rst_content = re.sub(r'^\s*(\w+(?:/\w+)?)$', lambda m: f'* [{m.group(1)}]({m.group(1)}.md)', rst_content, flags=re.MULTILINE)
    
    # Clean up extra whitespace
    rst_content = re.sub(r'\n{3,}', '\n\n', rst_content)
    
    return rst_content

def convert_file(input_path, output_path):
    """Convert a single RST file to Markdown."""
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    
    print(f"Converting {input_path} to {output_path}")
    
    # Check if input file exists
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}")
        return
    
    try:
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Read RST content
        with open(input_path, 'r', encoding='utf-8') as f:
            rst_content = f.read()
        
        # Convert to Markdown
        markdown_content = convert_rst_to_md(rst_content)
        
        # Write Markdown content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
            
        print(f"Successfully converted {input_path.name}")
            
    except Exception as e:
        print(f"Error converting {input_path}: {e}")

def main():
    """Main function to convert all RST files in the docs directory."""
    docs_dir = Path(__file__).parent.resolve()
    
    # Convert all RST files
    for rst_file in docs_dir.glob('**/*.rst'):
        md_file = rst_file.with_suffix('.md')
        convert_file(rst_file, md_file)
    
    print("Conversion complete!")

if __name__ == '__main__':
    main() 