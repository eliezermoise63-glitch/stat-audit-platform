# 🚀 Mise sur GitHub — Guide en quelques clics

Ce fichier vous guide pas à pas pour publier ce projet sur GitHub en moins de 5 minutes.

---

## Étape 1 — Créer le dépôt GitHub

1. Aller sur [github.com/new](https://github.com/new)
2. **Repository name** : `stat-audit-platform`
3. **Description** : `Statistical Audit Platform — ACP, Analyse Factorielle & LLM`
4. Choisir **Public** ou **Private**
5. ❌ Ne pas cocher "Add README" (vous en avez déjà un)
6. Cliquer **Create repository**

---

## Étape 2 — Initialiser et pousser depuis votre machine

Depuis le dossier `stat-audit-platform/` :

```bash
# Initialiser Git
git init

# Ajouter tous les fichiers (le .gitignore exclut secrets.toml automatiquement)
git add .

# Premier commit
git commit -m "feat: initial commit — Statistical Audit Platform v0.1.0"

# Lier au dépôt GitHub (remplacer YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/stat-audit-platform.git

# Pousser
git branch -M main
git push -u origin main
```

✅ Votre code est maintenant sur GitHub.

---

## Étape 3 — Déployer sur Streamlit Cloud (optionnel)

1. Aller sur [share.streamlit.io](https://share.streamlit.io)
2. Se connecter avec GitHub
3. Cliquer **New app**
4. Sélectionner :
   - **Repository** : `YOUR_USERNAME/stat-audit-platform`
   - **Branch** : `main`
   - **Main file path** : `app.py`
5. Cliquer **Advanced settings**
6. Dans l'onglet **Secrets**, coller :
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-votre-cle-ici"
   ```
7. Cliquer **Deploy** ✅

L'URL publique sera : `https://YOUR_USERNAME-stat-audit-platform-app-XXXX.streamlit.app`

---

## Étape 4 — Mettre à jour le README

Dans `README.md`, remplacer les deux occurrences de `YOUR_USERNAME` par votre vrai username GitHub.

```bash
# Sur Linux/macOS :
sed -i 's/YOUR_USERNAME/votre_vrai_username/g' README.md
git add README.md
git commit -m "docs: update username in README"
git push
```

---

## Structure des branches recommandée

```
main          ← code stable, déployé
  └── develop ← intégration des features
        └── feature/nom-de-la-feature
```

```bash
# Créer la branche develop
git checkout -b develop
git push -u origin develop
```

---

## Checklist avant de partager le lien

- [ ] `secrets.toml` absent du dépôt (vérifié par `.gitignore`)
- [ ] `README.md` mis à jour avec votre username
- [ ] CI GitHub Actions vert (badge visible dans le README)
- [ ] L'app Streamlit Cloud est accessible publiquement
- [ ] Le lien de l'app Streamlit est ajouté dans le README

---

*Ce projet fait partie de la plateforme Data Science personnelle d'Eliezer Moïse.*
