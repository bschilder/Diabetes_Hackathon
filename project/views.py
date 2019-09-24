from flask import render_template
from app import pages, server, freezer, app
# from app import server as app

@server.route('/dasher/')
def home():
    return render_template('index.html', dasher=app)

@freezer.register_generator
def product_details():
        return render_template('index.html', dasher=app)

@server.route('/<path:path>/')
def page(path):
    # `path` is the filename of a page, without the file extension
    # e.g. "first-post"
    page = pages.get_or_404(path)
    return render_template('page.html', page=page)
