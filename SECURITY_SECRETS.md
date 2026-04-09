# 🔐 Security Best Practices - Secrets Management

## ⚠️ Critical Security Rules

### ✅ DO
- ✅ Keep `.env` file **locally** with real secrets
- ✅ Add `.env` to `.gitignore` (already done in this project)
- ✅ Use `.env.example` as a **template reference** (no real secrets)
- ✅ Use GitHub Secrets for CI/CD deployments
- ✅ Rotate API keys regularly
- ✅ Use different keys for local dev vs production
- ✅ Review commits before pushing to ensure no secrets leaked

### ❌ DON'T
- ❌ **NEVER** commit `.env` file to git repository
- ❌ **NEVER** paste real API keys in documentation
- ❌ **NEVER** commit secrets to GitHub (even in history)
- ❌ **NEVER** hardcode secrets in source code
- ❌ **NEVER** share `.kubeconfig` files via unencrypted channels
- ❌ **NEVER** log sensitive data

---

## File Structure

```
c:\workspace\
├── .env                    # ← YOUR LOCAL SECRETS (never commit!)
│                           # ← Protected by .gitignore ✅
├── .env.example            # ← Template with placeholders (SAFE to commit)
├── template.env            # ← Alternative template
├── .gitignore              # ← Contains: .env, *.key, *.pem
│
├── k3s/
│   └── GITHUB_ACTIONS_SETUP.md     # ← Uses placeholders only
├── GITHUB_ACTIONS_DEPLOYMENT_READY.md  # ← Uses placeholders only
└── [all source code]       # ← No hardcoded secrets
```

---

## Workflow: Local Development

### 1. First Time Setup
```bash
# Copy template to real .env (with YOUR secrets)
cp .env.example .env

# Edit .env and add real values
nano .env
# HF_API_KEY=<your_token>
# MISTRAL_API_KEY=<your_key>
# etc.

# Verify .env is in .gitignore
grep "\.env" .gitignore     # Should show: .env

# Now source it (optional for local work)
source .env  # or: set -a; source .env; set +a
```

### 2. Before Any Git Push
```bash
# Verify NO secrets in staged files
git diff --cached | grep -i "api_key\|token\|secret"
# Should return: No matches

# Verify .env is NOT staged
git status
# Should NOT show: .env

# Safe to push
git push origin main
```

### 3. If Secrets Are Accidentally Committed
```bash
# STOP immediately - don't push!
# Remove from staging:
git reset HEAD <file>

# Remove from local commit:
git reset --soft HEAD~1

# IMPORTANT: If pushed to GitHub:
# 1. Rotate the leaked key immediately
# 2. Force push (if you own the repo)
# 3. Use GitHub's secret scanning alerts
```

---

## GitHub Actions: Secrets Management

### Create Secrets (One-Time Setup)
1. Go to: https://github.com/<owner>/<repo>/settings/secrets/actions
2. Create these secrets:

| Secret Name | Source |
|------------|--------|
| `K3S_KUBECONFIG` | Extract from datalab: `/etc/rancher/k3s/k3s.yaml` |
| `GHCR_TOKEN` | Generate: https://github.com/settings/tokens |
| `ENV_SECRETS` | Copy from your **local .env** (not from documentation!) |

### Access Secrets in Workflow
```yaml
# In .github/workflows/deploy-k3s.yml
- name: Setup Kubeconfig
  env:
    K3S_KUBECONFIG: ${{ secrets.K3S_KUBECONFIG }}  # ← GitHub Secret
    GHCR_TOKEN: ${{ secrets.GHCR_TOKEN }}          # ← GitHub Secret
    ENV_SECRETS: ${{ secrets.ENV_SECRETS }}        # ← GitHub Secret
  run: |
    # Secrets are available as environment variables
    # GitHub will automatically mask secrets in logs
```

### Important
- ✅ GitHub automatically **masks** secrets in logs
- ✅ Secrets are encrypted and stored securely
- ⚠️ If a secret is compromised, regenerate it immediately
- ⚠️ Never print secrets in workflows (GitHub catches some, not all)

---

## K3S Secrets Management

### Kubernetes Secrets (In-Cluster)
```bash
# Workflow creates these secrets automatically:

# 1. GHCR docker-registry secret
kubectl get secret ghcr-secret -n expert-immo
kubectl describe secret ghcr-secret -n expert-immo

# 2. Environment secrets
kubectl get secret expert-immo-env -n expert-immo
kubectl describe secret expert-immo-env -n expert-immo

# View secret value (base64 encoded, then decoded)
kubectl get secret expert-immo-env -n expert-immo -o jsonpath='{.data.HF_API_KEY}' | base64 -d
```

### Rotate Secrets in K3S
```bash
# Update GitHub Secret first:
# Settings → Secrets → Update ENV_SECRETS

# Then trigger workflow to redeploy:
git push origin main

# Or manually update K3S secret:
kubectl create secret generic expert-immo-env \
  --from-literal=HF_API_KEY=new_key \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart pods to pick up changes:
kubectl rollout restart deployment/mcp-server -n expert-immo
```

---

## Security Checklist

### Before Committing
- [ ] Run: `git diff HEAD | grep -i "key\|secret\|token\|password"`
- [ ] Should return: No matches
- [ ] Verify `.env` is in `.gitignore`
- [ ] Check `.gitignore` has: `.env`, `*.key`, `*.pem`, credentials
- [ ] Do NOT commit: `.env`, `kubeconfig`, SSL certificates, API keys

### GitHub Repository Settings
- [ ] Enable: "Require investigations" in branch protection
- [ ] Enable: GitHub Secret scanning (in Security tab)
- [ ] Review: Who has access to repository secrets
- [ ] Rotate: Tokens/keys every 90 days minimum

### Local Machine Security
- [ ] Encrypt `.env` file if shared drives
- [ ] Use: SSH keys instead of passwords for datalab
- [ ] Set: Restrictive file permissions: `chmod 600 .env`
- [ ] Use: .gitignore exclusively (not manual exclusion)

---

## Common Mistakes & How to Fix

### ❌ Mistake 1: Committed Secret in Git History
```bash
# If you committed a secret:
git log --all --oneline | grep -i "secret\|key"  # Find commit

# Option 1: Rewrite history (local only, not pushed)
git reset --soft HEAD~1
git reset HEAD <file>

# Option 2: If pushed to GitHub
# 1. IMMEDIATELY rotate the compromised key
# 2. Force push (dangerous, use with care)
# 3. Contact GitHub support to remove from backups
```

### ❌ Mistake 2: Secret in Documentation
**Example (BAD):**
```markdown
ENV_SECRETS=HF_API_KEY=sk-1234567890  ← DO NOT DO THIS!
```

**Example (GOOD):**
```markdown
ENV_SECRETS=HF_API_KEY=<your_token_from_local_env>  ← Use placeholders
```

### ❌ Mistake 3: .env Not in .gitignore
```bash
# Check if .env is ignored:
git check-ignore -v .env
# Should output: .gitignore:138:    .env

# If not ignored, add it:
echo ".env" >> .gitignore
git add .gitignore
git commit -m "chore: ensure .env is ignored"
```

---

## Emergency Procedures

### 🚨 Secret Leaked to Git/Internet
**Immediate Actions:**
1. **Stop everything** - Don't commit more code
2. **Rotate the key** - Generate new token/key immediately
3. **Notify team** - Alert anyone who might use it
4. **Remove from history**:
   ```bash
   git filter-branch --tree-filter 'rm -f leaked_file.txt'
   git push origin --force
   ```
5. **GitHub cleanup** - Secret may still be in GitHub cache/backups

### 🚨 Kubeconfig Compromised
**Immediate Actions:**
1. **Revoke in K3S**: Remove certificates from datalab
2. **Update K3S** - Regenerate certificates
3. **Rotate in GitHub** - Update K3S_KUBECONFIG secret
4. **Redeploy**: Trigger workflow to use new config

---

## Tools & Monitoring

### Prevent Secrets in Git (Pre-commit Hook)
```bash
# Install pre-commit framework
pip install pre-commit

# Create .pre-commit-config.yaml with secret scanner
cat > .pre-commit-config.yaml << 'EOF'
repos:
- repo: https://github.com/Yelp/detect-secrets
  rev: v1.4.0
  hooks:
  - id: detect-secrets
    args: ['--baseline', '.secrets.baseline']
EOF

# Install hooks
pre-commit install

# Now every commit will scan for secrets
```

### GitHub Secret Scanning
- ✅ Automatically enabled for all repositories
- ✅ Shows alerts if secrets are detected
- ✅ GitHub will notify you if your tokens are found in public code

See: https://github.com/settings/security

---

## References

- [GitHub: Managing Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [Kubernetes: Secrets](https://kubernetes.io/docs/concepts/configuration/secret/)
- [OWASP: Secrets Management](https://owasp.org/www-project-top-10/)
- [Detect Secrets GitHub](https://github.com/Yelp/detect-secrets)

---

**Remember:** Your CI/CD is only as secure as your secrets management! 🔒
