# 🎬 Rec

Rec is a web application allows users to discover and receive anime recommendations based on their preferences.

## ✨ Features
- 🔍 **Intuitive Search**: Real-time input with anime suggestions.
- 📊 **Personalized Recommendations**: Suggestions based on advanced criteria.
- 🌟 **Anime Exploration (Coming soon)**: Sort by popularity, score, and release year.
- 🔧 **Technologies**: Flask, PostgreSQL, Docker, and AI.

## 📚 Prerequisites
Before running the project, make sure you have:
- Python 3.10
- PostgreSQL installed and configured
- Docker and Docker Compose

## ♻️ Installation and Execution

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/rec-app.git
cd rec-app
```

### 2. Configure the Environment
Create a `.env` file at the project root and provide your variables:
```env
DATABASE_URL=postgresql://user:password@localhost/anime_db
FLASK_ENV=development
SECRET_KEY=your_secret_key
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the Application with Docker
```bash
docker-compose up --build
```
The application will be accessible at **http://localhost:5000**.

## 💡 Usage
1. **Search for an anime** in the search bar.
2. **Select up to 5 animes** to get recommendations.
3. **Explore the database** by sorting and filtering animes by score and date.

## 👨‍💻 Technologies Used
- **Backend**: Flask, SQLAlchemy
- **Database**: PostgreSQL
- **Machine Learning**: mlxtend, sentence-transformers
- **Frontend**: HTML, CSS, JavaScript
- **CI/CD**: GitHub Actions, Docker

## 🛠 Contribution
Contributions are welcome! To propose a modification:
1. Fork the project
2. Create a branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m "Added a new feature"`)
4. Push the branch (`git push origin feature-name`)
5. Open a Pull Request

## 🛡️ License
This project is licensed under the **MIT** license.

---

👤 **Author**: Youxise

