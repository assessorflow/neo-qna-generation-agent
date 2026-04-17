#!/usr/bin/env python3
"""Compile protobuf files for the project."""

import subprocess
import sys
from pathlib import Path


def compile_protos() -> int:
    """Compile all .proto files in the proto/ directory."""
    # Paths relative to project root
    project_root = Path(__file__).parent.parent
    proto_dir = project_root / "proto"
    stubs_dir = (
        project_root
        / "src"
        / "qna_generation_agent"
        / "infrastructure"
        / "grpc"
        / "stubs"
    )

    proto_files = list(proto_dir.rglob("*.proto"))

    if not proto_files:
        print("No .proto files found in proto/")
        return 1

    for proto_file in proto_files:
        print(f"Compiling {proto_file.relative_to(project_root)}...")

        # Calculate relative output path to maintain package structure
        relative_proto = proto_file.relative_to(proto_dir)
        output_subdir = stubs_dir / relative_proto.parent

        # Ensure output directory exists
        output_subdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"--proto_path={proto_dir}",
            f"--python_out={stubs_dir}",
            f"--grpc_python_out={stubs_dir}",
            str(proto_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error compiling {proto_file.name}:")
            print(result.stderr)
            return 1

        stem = proto_file.stem
        print(f"  ✓ Generated {relative_proto.parent}/{stem}_pb2.py")
        print(f"  ✓ Generated {relative_proto.parent}/{stem}_pb2_grpc.py")

        # Fix imports in generated grpc file to use absolute package paths
        grpc_file = output_subdir / f"{stem}_pb2_grpc.py"
        if grpc_file.exists():
            content = grpc_file.read_text()
            # Replace relative import with absolute package import
            # Original: from assessorflow.submission.v1 import submission_pb2 as ...
            # Fixed: from qna_generation_agent.infrastructure.grpc.stubs.assessorflow.submission.v1 import submission_pb2 as ...
            proto_package_path = relative_proto.parent.as_posix().replace("/", ".")
            old_import = f"from {proto_package_path} import {stem}_pb2 as"
            new_import = (
                f"from qna_generation_agent.infrastructure.grpc.stubs.{proto_package_path}"
                f" import {stem}_pb2 as"
            )
            content = content.replace(old_import, new_import)
            grpc_file.write_text(content)
            print(f"  ✓ Fixed imports in {grpc_file.name}")

    # Create __init__.py files for all subdirectories to make them Python packages
    for subdir in stubs_dir.rglob("*"):
        if subdir.is_dir():
            init_file = subdir / "__init__.py"
            if not init_file.exists():
                init_file.write_text('"""Generated gRPC stubs."""\n')
                print(f"  ✓ Created {init_file.relative_to(project_root)}")

    print("\nProto compilation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(compile_protos())
