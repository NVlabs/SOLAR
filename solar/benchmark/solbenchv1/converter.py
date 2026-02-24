#!/usr/bin/env python3
"""
Convert SolBench V1 benchmark files to Solar-compatible format.

Requirements from SOLBENCH_PLAN_v1.md:
- def get_inputs()
- ReferenceModel OR reference_backward
- launch_reference_implementation

This converter:
1. Checks if files have all required functions
2. If compliant → copies file as-is
3. If not compliant → modifies file to add required names, tracks changes
4. Outputs CSV of compliance status and changes made
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SolBenchV1Info:
    """Parsed information from a SolBench V1 file."""
    filename: str
    level: str
    index: str
    name: str
    has_get_inputs: bool
    has_reference_backward: bool
    has_reference_model: bool
    has_nn_module_class: bool
    nn_module_class_name: str
    has_launch_reference: bool
    source_code: str


class SolBenchV1Converter:
    """Convert SolBench V1 benchmark files to Solar format."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def parse_file(self, filepath: Path) -> SolBenchV1Info:
        """Parse a SolBench V1 file and extract information."""
        with open(filepath, 'r') as f:
            source_code = f.read()
        
        filename = filepath.name
        level = filepath.parent.name if filepath.parent.name.startswith('L') else "L1"
        
        match = re.match(r'^(\d+)_(.+)\.py$', filename)
        if match:
            index = match.group(1)
            name = match.group(2)
        else:
            index = "0000"
            name = filename.replace('.py', '')
        
        # Check for required functions
        has_get_inputs = 'def get_inputs(' in source_code
        has_reference_backward = 'def reference_backward(' in source_code
        has_reference_model = re.search(r'class\s+ReferenceModel\s*\(', source_code) is not None
        has_launch_reference = 'def launch_reference_implementation(' in source_code
        
        # Detect any nn.Module class (not ReferenceModel)
        nn_module_match = re.search(r'class\s+(\w+)\s*\(\s*nn\.Module\s*\)', source_code)
        has_nn_module_class = nn_module_match is not None
        nn_module_class_name = nn_module_match.group(1) if nn_module_match else ""
        
        # Don't count ReferenceModel as "other" nn.Module class
        if nn_module_class_name == "ReferenceModel":
            has_nn_module_class = False
            nn_module_class_name = ""
        
        return SolBenchV1Info(
            filename=filename,
            level=level,
            index=index,
            name=name,
            has_get_inputs=has_get_inputs,
            has_reference_backward=has_reference_backward,
            has_reference_model=has_reference_model,
            has_nn_module_class=has_nn_module_class,
            nn_module_class_name=nn_module_class_name,
            has_launch_reference=has_launch_reference,
            source_code=source_code,
        )
    
    def is_compliant(self, info: SolBenchV1Info) -> bool:
        """Check if file meets all requirements."""
        has_reference = info.has_reference_backward or info.has_reference_model
        return (info.has_get_inputs and 
                has_reference and 
                info.has_launch_reference)
    
    def fix_file(self, info: SolBenchV1Info) -> Tuple[str, List[str]]:
        """
        Fix a non-compliant file by renaming classes/functions.
        
        Returns:
            Tuple of (modified_code, list_of_changes)
        """
        code = info.source_code
        changes = []
        
        # If has nn.Module class but not ReferenceModel, rename it
        if info.has_nn_module_class and not info.has_reference_model and not info.has_reference_backward:
            old_name = info.nn_module_class_name
            # Rename class definition
            code = re.sub(
                rf'class\s+{old_name}\s*\(\s*nn\.Module\s*\)',
                'class ReferenceModel(nn.Module)',
                code
            )
            # Also rename usages in launch_reference_implementation if present
            if info.has_launch_reference:
                code = re.sub(
                    rf'\b{old_name}\b(?!\s*=)',  # Don't replace assignments
                    'ReferenceModel',
                    code
                )
            changes.append(f"renamed class {old_name} -> ReferenceModel")
        
        return code, changes
    
    def convert_directory(
        self, 
        input_dir: Path, 
        output_dir: Path,
        max_files: Optional[int] = None,
        csv_path: Optional[Path] = None
    ) -> Tuple[int, int, int, List[Dict[str, Any]]]:
        """
        Convert all SolBench V1 files in a directory.
        
        Returns:
            Tuple of (copied_count, modified_count, skipped_count, summary_records)
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        py_files = sorted(input_dir.glob("*.py"))
        
        if max_files:
            py_files = py_files[:max_files]
        
        copied = 0
        modified = 0
        skipped = 0
        summary_records = []
        
        for py_file in py_files:
            try:
                info = self.parse_file(py_file)
                
                # Check compliance
                has_reference = info.has_reference_backward or info.has_reference_model
                is_compliant = self.is_compliant(info)
                
                # Build record
                record = {
                    'filename': py_file.name,
                    'level': info.level,
                    'index': info.index,
                    'has_get_inputs': info.has_get_inputs,
                    'has_reference_backward': info.has_reference_backward,
                    'has_ReferenceModel': info.has_reference_model,
                    'has_other_nn_module': info.has_nn_module_class,
                    'other_nn_module_name': info.nn_module_class_name,
                    'has_launch_reference_implementation': info.has_launch_reference,
                    'is_compliant': is_compliant,
                    'action': '',
                    'changes': '',
                    'error': ''
                }
                
                # Output path maintains level structure
                output_file = output_dir / info.level / py_file.name
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Skip if missing get_inputs (can't fix)
                if not info.has_get_inputs:
                    record['action'] = 'SKIPPED'
                    record['error'] = 'missing get_inputs()'
                    summary_records.append(record)
                    skipped += 1
                    if self.debug:
                        print(f"SKIP: {py_file.name} - missing get_inputs()")
                    continue
                
                # Skip if no reference implementation at all
                if not has_reference and not info.has_nn_module_class:
                    record['action'] = 'SKIPPED'
                    record['error'] = 'missing reference implementation'
                    summary_records.append(record)
                    skipped += 1
                    if self.debug:
                        print(f"SKIP: {py_file.name} - no reference implementation")
                    continue
                
                if is_compliant:
                    # File is compliant - copy as-is
                    shutil.copy2(py_file, output_file)
                    record['action'] = 'COPIED'
                    copied += 1
                    if self.debug:
                        print(f"COPY: {py_file.name}")
                else:
                    # File needs modification
                    fixed_code, changes = self.fix_file(info)
                    
                    with open(output_file, 'w') as f:
                        f.write(fixed_code)
                    
                    record['action'] = 'MODIFIED'
                    record['changes'] = '; '.join(changes)
                    modified += 1
                    if self.debug:
                        print(f"MODIFY: {py_file.name} - {'; '.join(changes)}")
                
                summary_records.append(record)
                
            except Exception as e:
                record = {
                    'filename': py_file.name,
                    'level': input_dir.name,
                    'index': '',
                    'has_get_inputs': False,
                    'has_reference_backward': False,
                    'has_ReferenceModel': False,
                    'has_other_nn_module': False,
                    'other_nn_module_name': '',
                    'has_launch_reference_implementation': False,
                    'is_compliant': False,
                    'action': 'ERROR',
                    'changes': '',
                    'error': str(e)
                }
                summary_records.append(record)
                skipped += 1
                if self.debug:
                    print(f"ERROR: {py_file.name} - {e}")
        
        return copied, modified, skipped, summary_records
    
    def write_summary_csv(self, records: List[Dict[str, Any]], csv_path: Path):
        """Write summary records to CSV file."""
        import csv
        
        if not records:
            return
        
        fieldnames = [
            'filename', 'level', 'index',
            'has_get_inputs', 'has_reference_backward', 'has_ReferenceModel',
            'has_other_nn_module', 'other_nn_module_name',
            'has_launch_reference_implementation', 'is_compliant',
            'action', 'changes', 'error'
        ]
        
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)


def main():
    """CLI for converting SolBench V1 files."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert SolBench V1 benchmark files to Solar format'
    )
    parser.add_argument(
        '--input-dir', '-i',
        type=Path,
        required=True,
        help='Input directory containing benchmark files'
    )
    parser.add_argument(
        '--output-dir', '-o', 
        type=Path,
        required=True,
        help='Output directory for converted files'
    )
    parser.add_argument(
        '--max-files', '-n',
        type=int,
        default=None,
        help='Maximum number of files to convert'
    )
    parser.add_argument(
        '--csv', '-c',
        type=Path,
        default=None,
        help='Path to write summary CSV file'
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug output'
    )
    
    args = parser.parse_args()
    
    converter = SolBenchV1Converter(debug=args.debug)
    copied, modified, skipped, records = converter.convert_directory(
        args.input_dir, 
        args.output_dir,
        args.max_files
    )
    
    # Write CSV if path provided
    if args.csv:
        converter.write_summary_csv(records, args.csv)
        print(f"\nSummary CSV written to: {args.csv}")
    
    print(f"\nConversion complete:")
    print(f"  Copied (compliant):  {copied}")
    print(f"  Modified:            {modified}")
    print(f"  Skipped:             {skipped}")
    print(f"  Total:               {len(records)}")
    
    # Show modified files
    modified_records = [r for r in records if r.get('action') == 'MODIFIED']
    if modified_records:
        print(f"\nModified files ({len(modified_records)}):")
        for r in modified_records:
            print(f"  - {r['filename']}: {r['changes']}")


if __name__ == '__main__':
    main()
