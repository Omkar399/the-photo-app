"""CLI entry point for AFace."""

import sys
import click
from pathlib import Path

from the_photo_app.indexing.indexer import PhotoIndexer


@click.group()
def cli():
    """AFace - Local ultra-fast photo search."""
    pass


@cli.command()
@click.option(
    "--image-dir",
    type=click.Path(exists=True),
    default="data/images",
    help="Directory containing images to index",
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    help="Batch size for processing",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    default=True,
    help="Skip already indexed images",
)
def index(image_dir: str, batch_size: int, skip_existing: bool):
    """Index photos for semantic search."""
    click.echo("🖼️  AFace Indexer")
    click.echo(f"Image directory: {image_dir}")
    click.echo(f"Batch size: {batch_size}")

    indexer = PhotoIndexer()
    indexer.index_images(
        Path(image_dir),
        batch_size=batch_size,
        skip_existing=skip_existing,
    )

    click.echo("✅ Indexing complete!")


@cli.command()
@click.option(
    "--host",
    default="0.0.0.0",
    help="Server host",
)
@click.option(
    "--port",
    type=int,
    default=8000,
    help="Server port",
)
def serve(host: str, port: int):
    """Start FastAPI search server."""
    click.echo("🚀 Starting AFace server...")
    click.echo(f"Listening on {host}:{port}")

    from the_photo_app.api.server import main

    main()


@cli.command()
@click.option(
    "--port",
    type=int,
    default=8501,
    help="Streamlit port",
)
def ui(port: int):
    """Start Streamlit UI."""
    click.echo("🎨 Starting AFace UI...")
    click.echo(f"Launching Streamlit on port {port}")

    import subprocess

    subprocess.run(
        [
            "streamlit",
            "run",
            str(Path(__file__).parent / "ui" / "app.py"),
            f"--server.port={port}",
        ]
    )


if __name__ == "__main__":
    cli()
