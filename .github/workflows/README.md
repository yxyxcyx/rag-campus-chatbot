# CI/CD Pipeline Documentation

## Overview

This directory contains the GitHub Actions workflow for the MLOps pipeline.

## Workflow File

**`mlops-pipeline.yml`** - Main CI/CD pipeline with 7 stages:

1. **Code Quality** - Linting and formatting checks
2. **Unit Tests** - Automated testing with pytest
3. **RAG Evaluation** - Quality metrics using RAGAs framework
4. **Docker Build** - Multi-service container builds
5. **Integration Tests** - End-to-end system validation
6. **Security Scan** - Vulnerability scanning with Trivy
7. **Deployment Ready** - Final deployment validation

## Quick Setup

### Required Configuration

**Add GitHub Secret:**
```
Repository → Settings → Secrets and variables → Actions → New repository secret
Name: GROQ_API_KEY
Value: your_groq_api_key_here
```

### Optional Configuration

**Enable Container Registry (for Docker images):**
```
Repository → Settings → Packages → Connect repository
```

**Branch Protection Rules (recommended):**
```
Repository → Settings → Branches → Add rule for 'main'
- Require status checks: code-quality, rag-evaluation
- Require pull request reviews before merging
- Require conversation resolution before merging
```

## Pipeline Triggers

The pipeline runs automatically on:
- Push to `main` or `develop` branches
- Pull requests targeting `main` or `develop`
- Manual dispatch via GitHub UI

## Quality Gates

### Performance Thresholds

The pipeline enforces minimum quality standards:

| Metric | Threshold |
|--------|-----------|
| Context Precision | ≥ 0.70 |
| Context Recall | ≥ 0.70 |
| Faithfulness | ≥ 0.70 |
| Answer Relevancy | ≥ 0.70 |

**Enforcement:** Pipeline fails if any metric is below threshold, preventing deployment of low-quality models.

## Artifacts & Outputs

### Evaluation Results
- Location: Workflow artifacts (Actions tab)
- Retention: 30 days
- Format: CSV files with detailed metrics

### Docker Images
- Registry: `ghcr.io/<username>/rag-campus-chatbot-{api,worker,ui}`
- Tags: 
  - `latest` (main branch)
  - `<branch-name>` (feature branches)
  - `<branch>-<commit-sha>` (specific commits)

### Security Scans
- Location: Security tab → Code scanning alerts
- Scanner: Trivy
- Severity: Critical and High vulnerabilities

## Local Testing

Run the same checks locally before pushing:

```bash
# 1. Code quality
flake8 src/ scripts/
black --check src/ scripts/
isort --check-only src/ scripts/

# 2. Unit tests
pytest tests/ -v

# 3. RAG evaluation
python scripts/evaluate.py
python scripts/check_metrics.py

# 4. Docker builds
docker compose build
```

## Troubleshooting

### Pipeline Fails on Evaluation

**Problem:** `GROQ_API_KEY` not found or invalid, or "No evaluation results found"

**Solution:**
1. Add secret in GitHub repository settings (Settings → Secrets → Actions → New secret)
2. Verify API key is valid at https://console.groq.com/
3. Check key has sufficient quota
4. Ensure `eval_dataset.json` exists in repository root
5. If evaluation fails, pipeline will skip metrics check (safe for PRs)

### Docker Build Fails

**Problem:** Missing Dockerfile, build context issues, or "denied: installation not allowed to Create organization package"

**Solution:**
1. Verify all Dockerfiles exist: `Dockerfile.api`, `Dockerfile.worker`, `Dockerfile.ui`
2. Check `requirements/` directory has all necessary files
3. Review build logs for missing dependencies

**For "installation not allowed" error:**
- This is expected - images are **built but not pushed** by default
- To enable pushing to GitHub Container Registry, see main README CI/CD section
- The build still succeeds; only the push is disabled

### Quality Gates Failing

**Problem:** RAG metrics below thresholds

**Solution:**
1. Review evaluation results in artifacts
2. Check `eval_dataset.json` quality
3. Improve retrieval or generation prompts
4. Consider adjusting thresholds in `scripts/check_metrics.py` (if appropriate)

### Integration Tests Timeout

**Problem:** Services not starting in time

**Solution:**
1. Check Docker Compose configuration
2. Verify health check definitions
3. Increase timeout in workflow (currently 120s)
4. Review service logs in failed workflow

## Customization

### Modifying Thresholds

Edit `scripts/check_metrics.py`:

```python
THRESHOLDS = {
    'context_precision': 0.70,  # Adjust as needed
    'context_recall': 0.70,
    'faithfulness': 0.70,
    'answer_relevancy': 0.70
}
```

### Adding New Checks

1. Add step to existing job in `mlops-pipeline.yml`
2. Or create new job with dependencies
3. Update documentation accordingly

### Disabling Specific Jobs

Add conditional to job:

```yaml
job-name:
  if: github.event_name != 'pull_request'  # Skip on PRs
  # ... rest of job
```

## Best Practices

1. **Always run local tests** before pushing to main
2. **Keep evaluation dataset updated** with real-world queries
3. **Monitor artifact storage** (limited by GitHub plan)
4. **Review security scans regularly**
5. **Update dependencies** to patch vulnerabilities
6. **Use branch protection** to enforce quality gates

## Support

For issues with the CI/CD pipeline:
1. Check workflow run logs in Actions tab
2. Review this documentation
3. See main README.md and docs/DEVELOPMENT.md
4. Open GitHub issue with workflow run link
