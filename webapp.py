from flask import Flask, request, render_template, url_for, redirect
import pandas as pd
import json
app = Flask(__name__)
app.secret_key = 'You Will Never Guess'

import mrs

@app.route('/')
def recom():
    return render_template('app1.html')

@app.route('/pop_rating')
def pop_rating():
    app1 = mrs.qualified2
    print(app1)
    return render_template('pop-rating.html', app1=app1)

@app.route('/recom2', methods = ['GET','POST'])
def recom2():
    return render_template('app2.html')

@app.route('/genre_rating', methods = ['GET','POST'])
def genre_rating():
    if request.method == 'POST':
        genre = request.form['genre_choice']
        df = mrs.buildchart(genre)
        sf = pd.DataFrame(df)
        sf.to_csv('gen3', sep=',')
    return render_template('genre_rating.html', fram=sf )

@app.route('/movie_liked', methods = ['GET','POST'])
def movie_liked():
    return render_template('movie_liked.html')

@app.route('/movie_liked_recom', methods = ['GET','POST'])
def movies_liked_recom():
    if request.method == 'POST':
        title = request.form['title_choice']
        frame = mrs.improved_recommendations(title)
    return render_template('movie_liked_recom.html', frame=frame)

@app.route('/links')
def links():
    return render_template('links.html')

if __name__ == "__main__":
    app.run(debug=True)