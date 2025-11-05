from flask import Flask
from .config import Config
from .extension import db, migrate
# from .models import Coffee


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)

    with app.app_context():
        db.create_all()

    # Register blueprints or routes here if needed
    from .Routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app