# 1. Save a copy of your current project
cd C:\Users\Public\mkhalil\AI\AI

# 2. Remove old git history
rmdir /s /q .git

# 3. Reinitialize git
git init

# 4. Add your GitHub remote again
git remote add origin https://github.com/mmkhalil2002/AI.git

# 5. Add all your current (good) files
git add .

# 6. Commit them
git commit -m "Fresh clean commit"

# 7. Push fresh
git branch -M master
git push --force --set-upstream origin master
