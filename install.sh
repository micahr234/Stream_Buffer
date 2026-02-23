#!/bin/bash
# Installation script for project dependencies
# Run with: source install.sh


# Logging function
log() {
    echo "[INFO] $1"
}

warn() {
    echo "[WARN] $1"
}

error() {
    echo "[ERROR] $1"
}

success() {
    echo "[SUCCESS] $1"
}

# Setup git configuration
setup_git() {
    log "Setting up git configuration..."
    
    git config --global user.email "user@example.com"
    git config --global user.name "User"
    success "Git configured: User <user@example.com>"s
}

# Install uv package manager
install_uv() {
    log "Installing uv package manager..."
    
    # Check if uv is already installed
    if command -v uv >/dev/null 2>&1; then
        success "uv is already installed: $(uv --version)"
        return
    fi
    
    # Download and install uv
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        error "Failed to install uv."
        exit 1
    fi
    
    # Add to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    # Verify installation
    if ! command -v uv >/dev/null 2>&1; then
        error "Failed to install uv."
        exit 1
    fi
    
    success "uv installed successfully: $(uv --version)"
}

# Create and setup virtual environment
setup_venv() {
    log "Creating virtual environment..."
    
    # Remove existing venv if it exists
    if [ -d ".venv" ]; then
        warn "Removing existing virtual environment..."
        rm -rf .venv
    fi
    
    # Create new virtual environment
    if ! uv venv; then
        error "Failed to create virtual environment"
        exit 1
    fi
    
    success "Virtual environment created"
    
    # Install project dependencies
    log "Installing project dependencies..."
    if ! uv pip install -e ".[dev]" --python .venv/bin/python --prerelease=allow; then
        error "Failed to install project dependencies"
        exit 1
    fi
    
    # Install PyTorch with CUDA support if CUDA is available
    if command -v nvidia-smi >/dev/null 2>&1; then
        log "CUDA detected. Installing PyTorch with CUDA support..."
        # Use --index-url (deprecated but still works) or --index for newer uv versions
        if ! uv pip install torch --index-url https://download.pytorch.org/whl/cu131 --python .venv/bin/python; then
            warn "Failed to install PyTorch with CUDA support. Falling back to CPU version."
        else
            success "PyTorch with CUDA support installed in virtual environment"
        fi
    else
        warn "CUDA not detected. PyTorch will use CPU only."
    fi
    
    success "Project dependencies installed"
}

# List YAML config files in configs/ (one launch entry per file)
find_config_yamls() {
    [ ! -d configs ] && return
    for f in configs/*.yaml configs/*.yml; do
        [ -f "$f" ] && echo "$f"
    done | sort
}

determine_script_type() {
    echo "scripts/run.py"
}

generate_display_name() {
    local filename="$1"
    local basename=$(basename "$filename" .yaml)
    basename=$(basename "$basename" .yml)
    echo "$basename" | sed 's/_/ /g' | sed 's/\b\w/\U&/g'
}

# Setup VS Code configuration for remote development
# Auto-writes .vscode/launch.json from configs/*.yaml. settings.json and pyrightconfig.json persist from repo.
setup_vscode() {
    log "Setting up VS Code configuration..."
    mkdir -p .vscode

    local yaml_files
    mapfile -t yaml_files < <(find_config_yamls)

    if [ ${#yaml_files[@]} -eq 0 ]; then
        warn "No YAML files found; skipping launch.json generation"
    else
        log "Writing .vscode/launch.json for ${#yaml_files[@]} config(s): ${yaml_files[*]}"
        cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
EOF
        for yaml_file in "${yaml_files[@]}"; do
            local script_path=$(determine_script_type "$yaml_file")
            local display_name=$(generate_display_name "$yaml_file")
            cat >> .vscode/launch.json << EOF
        {
            "name": "$display_name",
            "type": "debugpy",
            "request": "launch",
            "program": "\${workspaceFolder}/$script_path",
            "console": "integratedTerminal",
            "args": ["$yaml_file"],
            "python": "\${workspaceFolder}/.venv/bin/python",
            "cwd": "\${workspaceFolder}",
            "env": {
                "PYTHONPATH": "\${workspaceFolder}/src"
            }
        }
EOF
            [ "$yaml_file" != "${yaml_files[-1]}" ] && echo "," >> .vscode/launch.json
        done
        cat >> .vscode/launch.json << 'EOF'
    ]
}
EOF
        success "Wrote .vscode/launch.json"
    fi
}

# Main installation process
main() {
    echo "Starting Installation"
    echo "=================================="
    
    # Run installation steps
    setup_git
    install_uv
    setup_venv
    
    echo ""
    echo "Installation complete!"
    
    # VS Code: extensions recommendations + launch.json from configs/*.yaml
    log "Setting up VS Code..."
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
    setup_vscode
}

# Run main function
main "$@"
