⚡ NeutronRAG
C++ Accelerated Retrieval-Augmented Generation

Deployed APP : https://neutronrag.streamlit.app/

NeutronRAG is a high-performance system designed for document intelligence. It swaps traditional Python-based search for a custom C++ Similarity Engine to achieve faster, more efficient data retrieval

🚀 Key Features
C++ Backend: Uses a compiled neutron_math module for high-speed vector matching (similarity)

Intelligent Build: Automatically compiles C++ source code to match the host environment (Mac or Cloud)

AI Powered: Leverages Gemini 2.5 Flash for rapid, context-grounded answers

Seamless PDF Indexing: Processes and chunks PDFs using PyMuPDF and sentence-transformers

🛠️ Technical Stack
Language: Python & C++11

AI: Google Gemini API & Sentence-Transformers

Framework: Streamlit

Interface: pybind11

🛠️ Technical Challenges & Solutions
Cross-Platform Compilation: Solved by using sysconfig to dynamically locate Python headers, allowing the same code to build on both macOS and Linux

Environment Parity: Resolved "Invalid ELF header" errors by ensuring C++ binaries are built natively on the target server rather than transferred from local machines

Dependency Management: Configured packages.txt to provide essential build tools (build-essential) missing from standard Python environments

Version Compatibility: Dynamically mapped include paths to support the latest Python 3.13 runtime
