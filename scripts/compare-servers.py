#!/usr/bin/env python3
"""
Compare embellama server against llama-server for embedding compatibility.

This script:
1. Starts both llama-server and embellama-server automatically
2. Runs a comprehensive test suite with various text inputs
3. Compares embeddings using cosine similarity and L2 distance
4. Reports results with colored output
5. Cleans up both servers on exit

Usage:
    python compare-servers.py /path/to/model.gguf [--threshold 0.99] [--workers 2]
"""

import sys
import argparse
import subprocess
import time
import signal
import atexit
import socket
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

try:
    from openai import OpenAI
except ImportError:
    print("Error: OpenAI Python SDK not installed.")
    print("Please install it with: pip install openai")
    sys.exit(1)


# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


@dataclass
class ServerProcess:
    """Manages a server process lifecycle"""
    name: str
    process: subprocess.Popen
    url: str
    port: int

    def is_healthy(self) -> bool:
        """Check if server is responding"""
        client = OpenAI(base_url=f"{self.url}/v1", api_key="dummy")
        try:
            client.models.list()
            return True
        except:
            return False

    def shutdown(self):
        """Gracefully shutdown the server"""
        if self.process:
            print(f"Shutting down {self.name}...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


@dataclass
class ComparisonResult:
    """Result of comparing two embeddings"""
    test_name: str
    cosine_similarity: float
    l2_distance: float
    max_diff: float
    llama_time: float
    embellama_time: float
    passed: bool
    error: Optional[str] = None


class ServerManager:
    """Manages both llama-server and embellama-server lifecycle"""

    def __init__(self, model_path: str, workers: int = 2):
        self.model_path = model_path
        self.workers = workers
        self.servers: List[ServerProcess] = []
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\nReceived interrupt signal, cleaning up...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        """Shutdown all servers"""
        for server in self.servers:
            server.shutdown()
        self.servers.clear()

    def _find_available_port(self, start_port: int) -> int:
        """Find an available port starting from start_port"""
        port = start_port
        while port < start_port + 100:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('127.0.0.1', port))
                sock.close()
                return port
            except OSError:
                port += 1
        raise RuntimeError(f"No available port found in range {start_port}-{start_port + 100}")

    def start_llama_server(self) -> ServerProcess:
        """Start llama-server with specified configuration"""
        port = self._find_available_port(8080)
        print(f"{Colors.OKCYAN}Starting llama-server on port {port}...{Colors.ENDC}")

        cmd = [
            "llama-server",
            "--embedding",
            "-m", self.model_path,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--ctx-size", "32768",
            "--ubatch-size", "8192",
            "--pooling", "last",
            "-v"
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        server = ServerProcess(
            name="llama-server",
            process=process,
            url=f"http://127.0.0.1:{port}",
            port=port
        )

        self.servers.append(server)
        return server

    def start_embellama_server(self) -> ServerProcess:
        """Start embellama-server with specified configuration"""
        port = self._find_available_port(8081)
        print(f"{Colors.OKCYAN}Starting embellama-server on port {port}...{Colors.ENDC}")

        cmd = [
            "cargo", "run", "--release", "--features", "server", "--bin", "embellama-server", "--",
            "--model-path", self.model_path,
            "--model-name", "test-model",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--workers", str(self.workers),
            "--pooling-strategy", "last",  # Match llama-server's --pooling last
            "--log-level", "info"
        ]

        # Set context size via environment variable
        env = {"EMBELLAMA_TEST_CONTEXT_SIZE": "32768"}

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**subprocess.os.environ, **env}
        )

        server = ServerProcess(
            name="embellama-server",
            process=process,
            url=f"http://127.0.0.1:{port}",
            port=port
        )

        self.servers.append(server)
        return server

    def wait_for_health(self, server: ServerProcess, timeout: int = 60) -> bool:
        """Wait for server to become healthy"""
        print(f"Waiting for {server.name} to be ready...", end="", flush=True)
        start = time.time()

        while time.time() - start < timeout:
            if server.process.poll() is not None:
                print(f" {Colors.FAIL}FAILED (process died){Colors.ENDC}")
                return False

            if server.is_healthy():
                print(f" {Colors.OKGREEN}OK{Colors.ENDC}")
                return True

            print(".", end="", flush=True)
            time.sleep(0.5)

        print(f" {Colors.FAIL}TIMEOUT{Colors.ENDC}")
        return False


class EmbeddingComparator:
    """Compares embeddings from two servers"""

    def __init__(self, llama_url: str, embellama_url: str, threshold: float = 0.99):
        self.llama_client = OpenAI(base_url=f"{llama_url}/v1", api_key="dummy")
        self.embellama_client = OpenAI(base_url=f"{embellama_url}/v1", api_key="dummy")
        self.threshold = threshold

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

    @staticmethod
    def l2_distance(a: List[float], b: List[float]) -> float:
        """Calculate L2 distance between two vectors"""
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.linalg.norm(a_arr - b_arr))

    @staticmethod
    def max_diff(a: List[float], b: List[float]) -> float:
        """Calculate maximum element-wise difference"""
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.max(np.abs(a_arr - b_arr)))

    def compare_single(self, test_name: str, text: str) -> ComparisonResult:
        """Compare embeddings for a single text"""
        try:
            # Get embedding from llama-server
            start = time.time()
            llama_response = self.llama_client.embeddings.create(
                model="test-model",
                input=text
            )
            llama_time = time.time() - start
            llama_embedding = llama_response.data[0].embedding

            # Get embedding from embellama-server
            start = time.time()
            embellama_response = self.embellama_client.embeddings.create(
                model="test-model",
                input=text
            )
            embellama_time = time.time() - start
            embellama_embedding = embellama_response.data[0].embedding

            # Validate dimensions match
            if len(llama_embedding) != len(embellama_embedding):
                return ComparisonResult(
                    test_name=test_name,
                    cosine_similarity=0.0,
                    l2_distance=float('inf'),
                    max_diff=float('inf'),
                    llama_time=llama_time,
                    embellama_time=embellama_time,
                    passed=False,
                    error=f"Dimension mismatch: llama={len(llama_embedding)}, embellama={len(embellama_embedding)}"
                )

            # Calculate metrics
            cos_sim = self.cosine_similarity(llama_embedding, embellama_embedding)
            l2_dist = self.l2_distance(llama_embedding, embellama_embedding)
            max_d = self.max_diff(llama_embedding, embellama_embedding)

            passed = cos_sim >= self.threshold

            return ComparisonResult(
                test_name=test_name,
                cosine_similarity=cos_sim,
                l2_distance=l2_dist,
                max_diff=max_d,
                llama_time=llama_time,
                embellama_time=embellama_time,
                passed=passed
            )

        except Exception as e:
            return ComparisonResult(
                test_name=test_name,
                cosine_similarity=0.0,
                l2_distance=float('inf'),
                max_diff=float('inf'),
                llama_time=0.0,
                embellama_time=0.0,
                passed=False,
                error=str(e)
            )

    def compare_batch(self, test_name: str, texts: List[str]) -> ComparisonResult:
        """Compare embeddings for a batch of texts"""
        try:
            # Get embeddings from llama-server
            start = time.time()
            llama_response = self.llama_client.embeddings.create(
                model="test-model",
                input=texts
            )
            llama_time = time.time() - start

            # Get embeddings from embellama-server
            start = time.time()
            embellama_response = self.embellama_client.embeddings.create(
                model="test-model",
                input=texts
            )
            embellama_time = time.time() - start

            # Validate batch sizes match
            if len(llama_response.data) != len(embellama_response.data):
                return ComparisonResult(
                    test_name=test_name,
                    cosine_similarity=0.0,
                    l2_distance=float('inf'),
                    max_diff=float('inf'),
                    llama_time=llama_time,
                    embellama_time=embellama_time,
                    passed=False,
                    error=f"Batch size mismatch: llama={len(llama_response.data)}, embellama={len(embellama_response.data)}"
                )

            # Compare each embedding in the batch
            similarities = []
            distances = []
            diffs = []

            for i in range(len(texts)):
                llama_emb = llama_response.data[i].embedding
                embellama_emb = embellama_response.data[i].embedding

                if len(llama_emb) != len(embellama_emb):
                    return ComparisonResult(
                        test_name=test_name,
                        cosine_similarity=0.0,
                        l2_distance=float('inf'),
                        max_diff=float('inf'),
                        llama_time=llama_time,
                        embellama_time=embellama_time,
                        passed=False,
                        error=f"Dimension mismatch at index {i}: llama={len(llama_emb)}, embellama={len(embellama_emb)}"
                    )

                similarities.append(self.cosine_similarity(llama_emb, embellama_emb))
                distances.append(self.l2_distance(llama_emb, embellama_emb))
                diffs.append(self.max_diff(llama_emb, embellama_emb))

            # Use average metrics for batch
            avg_cos_sim = float(np.mean(similarities))
            avg_l2_dist = float(np.mean(distances))
            avg_max_diff = float(np.mean(diffs))

            passed = avg_cos_sim >= self.threshold

            return ComparisonResult(
                test_name=test_name,
                cosine_similarity=avg_cos_sim,
                l2_distance=avg_l2_dist,
                max_diff=avg_max_diff,
                llama_time=llama_time,
                embellama_time=embellama_time,
                passed=passed
            )

        except Exception as e:
            return ComparisonResult(
                test_name=test_name,
                cosine_similarity=0.0,
                l2_distance=float('inf'),
                max_diff=float('inf'),
                llama_time=0.0,
                embellama_time=0.0,
                passed=False,
                error=str(e)
            )


def get_test_sequences() -> Dict[str, List[str]]:
    """Define test sequences for comparison"""
    return {
        "short_texts": [
            "Hello world",
            "Test embedding",
            "Quick test"
        ],
        "medium_texts": [
            "This is a medium length text that contains multiple sentences. "
            "It's designed to test how the embedding systems handle typical paragraph-sized inputs. "
            "The text should be processed correctly and return similar embeddings from both servers.",

            "Machine learning models are transforming how we process and understand natural language. "
            "Embeddings capture semantic meaning in high-dimensional vector spaces, enabling "
            "similarity comparisons and semantic search applications.",
        ],
        "long_texts": [
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 20,

            "Natural language processing has evolved significantly over the past decade. "
            "Modern transformer architectures have revolutionized how we approach text understanding. "
            "Embeddings learned from large corpora capture nuanced semantic relationships. " * 10
        ],
        "code_snippets": [
            "def hello_world():\n    print('Hello, world!')\n    return 0",

            "function fibonacci(n) {\n"
            "    if (n <= 1) return n;\n"
            "    return fibonacci(n - 1) + fibonacci(n - 2);\n"
            "}",

            "class Example {\n"
            "    constructor(value) {\n"
            "        this.value = value;\n"
            "    }\n"
            "}"
        ],
        "special_chars": [
            "Text with special characters: café, naïve, 日本語, emoji 🚀",
            "Symbols and math: α + β = γ, ∑∞ₙ₌₁ 1/n²",
            "Mixed: Hello世界! Test@#$%^&*()"
        ],
        "batch_small": [
            "First text in batch",
            "Second text in batch",
            "Third text in batch"
        ],
        "batch_medium": [
            f"Batch text number {i}" for i in range(10)
        ],
        "batch_large": [
            f"Large batch item {i} with some additional content to make it more realistic"
            for i in range(25)
        ]
    }


def print_result(result: ComparisonResult):
    """Print a comparison result with color coding"""
    status = f"{Colors.OKGREEN}PASS{Colors.ENDC}" if result.passed else f"{Colors.FAIL}FAIL{Colors.ENDC}"

    print(f"\n  {Colors.BOLD}{result.test_name}{Colors.ENDC}: {status}")

    if result.error:
        print(f"    {Colors.FAIL}Error: {result.error}{Colors.ENDC}")
    else:
        print(f"    Cosine similarity: {result.cosine_similarity:.6f}")
        print(f"    L2 distance: {result.l2_distance:.6f}")
        print(f"    Max difference: {result.max_diff:.6f}")
        print(f"    Response time: llama={result.llama_time:.3f}s, embellama={result.embellama_time:.3f}s")


def print_summary(results: List[ComparisonResult], threshold: float):
    """Print summary statistics"""
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)

    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}SUMMARY{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")

    print(f"Total tests: {total}")
    print(f"{Colors.OKGREEN}Passed: {passed}{Colors.ENDC}")
    print(f"{Colors.FAIL}Failed: {failed}{Colors.ENDC}")
    print(f"Success rate: {100 * passed / total:.1f}%")
    print(f"Similarity threshold: {threshold:.4f}")

    if results:
        valid_results = [r for r in results if not r.error]
        if valid_results:
            avg_cosine = np.mean([r.cosine_similarity for r in valid_results])
            avg_l2 = np.mean([r.l2_distance for r in valid_results])
            avg_llama_time = np.mean([r.llama_time for r in valid_results])
            avg_embellama_time = np.mean([r.embellama_time for r in valid_results])

            print(f"\nAverage cosine similarity: {avg_cosine:.6f}")
            print(f"Average L2 distance: {avg_l2:.6f}")
            print(f"Average response time: llama={avg_llama_time:.3f}s, embellama={avg_embellama_time:.3f}s")

    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare embellama-server against llama-server for embedding compatibility"
    )
    parser.add_argument(
        "model_path",
        help="Path to GGUF model file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="Cosine similarity threshold for passing (default: 0.99)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of workers for embellama-server (default: 2)"
    )

    args = parser.parse_args()

    print(f"\n{Colors.HEADER}{Colors.BOLD}Embellama vs llama-server Comparison{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")
    print(f"Model: {args.model_path}")
    print(f"Threshold: {args.threshold}")
    print(f"Workers: {args.workers}\n")

    # Start servers
    manager = ServerManager(args.model_path, args.workers)

    try:
        llama_server = manager.start_llama_server()
        if not manager.wait_for_health(llama_server):
            print(f"{Colors.FAIL}Failed to start llama-server{Colors.ENDC}")
            return 1

        embellama_server = manager.start_embellama_server()
        if not manager.wait_for_health(embellama_server):
            print(f"{Colors.FAIL}Failed to start embellama-server{Colors.ENDC}")
            return 1

        print(f"\n{Colors.OKGREEN}Both servers are ready!{Colors.ENDC}\n")

        # Create comparator
        comparator = EmbeddingComparator(
            llama_server.url,
            embellama_server.url,
            args.threshold
        )

        # Run tests
        test_sequences = get_test_sequences()
        results = []

        print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}RUNNING TESTS{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")

        # Single text tests
        for category, texts in test_sequences.items():
            if category.startswith("batch"):
                # Handle batch tests separately
                result = comparator.compare_batch(category, texts)
                results.append(result)
                print_result(result)
            else:
                # Single text tests
                for i, text in enumerate(texts):
                    test_name = f"{category}[{i}]"
                    result = comparator.compare_single(test_name, text)
                    results.append(result)
                    print_result(result)

        # Print summary
        print_summary(results, args.threshold)

        # Return exit code based on results
        return 0 if all(r.passed for r in results) else 1

    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Interrupted by user{Colors.ENDC}")
        return 130
    except Exception as e:
        print(f"\n{Colors.FAIL}Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        manager.cleanup()


if __name__ == "__main__":
    sys.exit(main())
