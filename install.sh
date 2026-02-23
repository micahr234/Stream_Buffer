#!/bin/bash
# Installation script for project dependencies
# Run with: source install.sh


log() { echo "[INFO] $1"; }
warn() { echo "[WARN] $1"; }
error() { echo "[ERROR] $1"; }
success() { echo "[SUCCESS] $1"; }


setup_git() {
    log "Setting up git configuration..."
    git config --global user.email "user@example.com"
    git config --global user.name "User"
    success "Git configured: User <user@example.com>"
}


install_uv() {
    log "Installing uv package manager..."
    if command -v uv >/dev/null 2>&1; then
        success "uv is already installed: $(uv --version)"
        return
    fi
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        error "Failed to install uv."
        exit 1
    fi
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        error "Failed to install uv."
        exit 1
    fi
    success "uv installed successfully: $(uv --version)"
}


setup_venv() {
    log "Creating virtual environment..."
    if [ -d ".venv" ]; then
        warn "Removing existing virtual environment..."
        rm -rf .venv
    fi
    if ! uv venv; then
        error "Failed to create virtual environment"
        exit 1
    fi
    success "Virtual environment created"

    log "Installing project dependencies..."
    if ! uv pip install -e ".[dev]" --python .venv/bin/python --prerelease=allow; then
        error "Failed to install project dependencies"
        exit 1
    fi
    success "Project dependencies installed"
}


setup_vscode() {
    log "Setting up VS Code configuration..."
    mkdir -p .vscode
    cat > .vscode/extensions.json << 'EOF'
{
    "recommendations": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-python.debugpy"
    ]
}
EOF
    success "Wrote .vscode/extensions.json"
}


main() {
    echo "Starting Installation"
    echo "=================================="
    setup_git
    install_uv
    setup_venv
    setup_vscode
    echo ""
    echo "Installation complete!"
}

main "$@"
