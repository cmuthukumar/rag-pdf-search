# GitHub Setup Instructions

## Create a New Repository on GitHub

1. Go to [GitHub](https://github.com) and log in
2. Click the "+" icon in the top right → "New repository"
3. Fill in the details:
   - **Repository name**: `rag-pdf-search` (or your preferred name)
   - **Description**: "RAG system for semantic search over PDF documents from S3"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Click "Create repository"

## Connect Local Repository to GitHub

After creating the repository on GitHub, run these commands:

```bash
# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/rag-pdf-search.git

# Verify the remote was added
git remote -v

# Push to GitHub
git push -u origin main
```

## Alternative: Using SSH

If you prefer SSH:

```bash
# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin git@github.com:YOUR_USERNAME/rag-pdf-search.git

# Push to GitHub
git push -u origin main
```

## Making Future Changes

After making changes to your code:

```bash
# Check what files changed
git status

# Stage changes
git add .

# Commit with a descriptive message
git commit -m "Your descriptive commit message"

# Push to GitHub
git push
```

## Common Git Commands

```bash
# View commit history
git log --oneline

# Create a new branch
git checkout -b feature-name

# Switch branches
git checkout main

# Merge a branch
git merge feature-name

# Pull latest changes
git pull

# View changes before committing
git diff
```

## GitHub Repository Settings

### Recommended Settings

1. **Add topics**: `rag`, `langchain`, `chromadb`, `pdf-processing`, `semantic-search`, `python`
2. **Add a description**: Short summary of what the project does
3. **Enable Issues**: For tracking bugs and feature requests
4. **Enable Discussions**: For Q&A and community interaction

### Branch Protection (Optional)

For collaborative projects:
1. Go to Settings → Branches
2. Add rule for `main` branch
3. Enable:
   - Require pull request reviews
   - Require status checks to pass

## Sharing Your Repository

Your repository will be available at:
```
https://github.com/YOUR_USERNAME/rag-pdf-search
```

Share this URL with collaborators or include it in your portfolio!
