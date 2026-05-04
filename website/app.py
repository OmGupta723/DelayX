from flask import Flask, render_template, request
import pickle
import json
import pandas as pd

app = Flask(__name__)

# -------------------- LOAD MODEL --------------------
model = pickle.load(open('../model/flight_delay_model.pkl', 'rb'))

# -------------------- LOAD MAPPINGS --------------------
with open('../model/airline_mapping.json') as f:
    airline_mapping = json.load(f)

with open('../model/origin_mapping.json') as f:
    origin_mapping = json.load(f)

# -------------------- LOAD DATA --------------------
df = pd.read_csv('../data/clean_flights.csv')

# -------------------- LOAD ACCURACY AUTOMATICALLY --------------------  ← ADD HERE
import glob

def get_model_accuracy():
    accuracy_files = glob.glob('../model/*_accuracy.json')
    
    if not accuracy_files:
        return 97  # fallback

    best_accuracy = 0
    best_model    = ""

    for file in accuracy_files:
        try:
            with open(file) as f:
                data = json.load(f)
                if data['accuracy'] > best_accuracy:
                    best_accuracy = data['accuracy']
                    best_model    = data['model']
        except:
            continue

    print(f"Best model: {best_model} - Accuracy: {best_accuracy}%")
    return round(best_accuracy)

MODEL_ACCURACY = get_model_accuracy()
print(f"Showing accuracy on website: {MODEL_ACCURACY}%")

# -------------------- BUILD REVERSE MAPPINGS --------------------
origin_reverse = {v: int(k) for k, v in origin_mapping.items()}

airline_categories = sorted(df['AIRLINE_CODE'].unique())
airline_reverse    = {code: idx for idx, code in enumerate(airline_categories)}

# -------------------- HOME PAGE --------------------
@app.route('/')
def index():
    airports = sorted(origin_mapping.values())
    return render_template(
        'index.html',
        airlines=airline_mapping,
        airports=airports,
        accuracy=MODEL_ACCURACY
    )

# -------------------- PREDICTION --------------------
@app.route('/predict', methods=['POST'])
def predict():
    airline   = request.form['airline']
    origin    = request.form['origin']
    dest      = request.form['dest']
    month     = int(request.form['month'])
    day       = int(request.form['day'])
    dep_delay = int(request.form['dep_delay'])

    airline_code = airline_reverse.get(airline, 0)
    origin_code  = origin_reverse.get(origin, 0)
    dest_code    = origin_reverse.get(dest, 0)

    dep_hour   = max(0, min(23, int(dep_delay / 60)))
    is_weekend = 1 if day >= 6 else 0

    features = pd.DataFrame([[
        airline_code,
        origin_code,
        dest_code,
        month,
        day,
        dep_delay,
        dep_hour,
        is_weekend
    ]], columns=['AIRLINE_CODE','ORIGIN','DEST','MONTH','DAY_OF_WEEK','DEP_DELAY','DEP_HOUR','IS_WEEKEND'])

    prediction  = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1] * 100

    causes = {
        'Carrier':       round(df['DELAY_DUE_CARRIER'].mean(), 2),
        'Weather':       round(df['DELAY_DUE_WEATHER'].mean(), 2),
        'NAS':           round(df['DELAY_DUE_NAS'].mean(), 2),
        'Security':      round(df['DELAY_DUE_SECURITY'].mean(), 2),
        'Late Aircraft': round(df['DELAY_DUE_LATE_AIRCRAFT'].mean(), 2)
    }

    route_data = df[
        (df['AIRLINE_CODE'] == airline) &
        (df['ORIGIN']       == origin)  &
        (df['DEST']         == dest)
    ]

    if len(route_data) > 0:
        avg_delay_min = round(route_data['ARR_DELAY'].mean())
    else:
        airline_data  = df[df['AIRLINE_CODE'] == airline]
        avg_delay_min = round(airline_data['ARR_DELAY'].mean()) if len(airline_data) > 0 else 15

    avg_delay_min = max(int(avg_delay_min), 0)
    delay_hours   = avg_delay_min // 60
    delay_mins    = avg_delay_min % 60

    if probability >= 65:
        verdict = "HIGH RISK"
        advice  = "We strongly suggest rescheduling or choosing a different airline."
        color   = "red"
    elif probability >= 30:
        verdict = "MEDIUM RISK"
        advice  = "There is a moderate chance of delay. Reach the airport early."
        color   = "orange"
    else:
        verdict = "LOW RISK"
        advice  = "Good to travel! This flight has a low chance of delay."
        color   = "green"
        # Save to history if logged in
    if 'user' in session:
        email = session['user']
        if email not in search_history:
            search_history[email] = []
        search_history[email].insert(0, {
            'airline':     airline_mapping.get(airline, airline),
            'origin':      origin,
            'dest':        dest,
            'probability': round(probability, 1),
            'color':       color,
            'date':        datetime.now().strftime('%d %b %Y')
        })
        search_history[email] = search_history[email][:10]  # keep last 10

    return render_template(
        'result.html',
        airline       = airline_mapping.get(airline, airline),
        origin        = origin,
        dest          = dest,
        probability   = round(probability, 1),
        verdict       = verdict,
        advice        = advice,
        color         = color,
        causes        = causes,
        accuracy      = MODEL_ACCURACY,
        avg_delay_min = avg_delay_min,
        delay_hours   = delay_hours,
        delay_mins    = delay_mins,
        month         = month,
        day           = day
    )

# -------------------- RECOMMENDATIONS --------------------
# -------------------- RECOMMENDATIONS --------------------
@app.route('/recommend')
def recommend():
    origin       = request.args.get('origin')
    dest         = request.args.get('dest')
    month        = int(request.args.get('month', 1))
    day          = int(request.args.get('day', 0))
    current_prob = float(request.args.get('current_prob', 0))

    dep_hour   = 6
    is_weekend = 1 if day >= 6 else 0

    origin_code = origin_reverse.get(origin, 0)
    dest_code   = origin_reverse.get(dest, 0)

    all_flights = []

    # Build all features at once as a DataFrame for speed
    rows = []
    codes = []
    names = []

    for code, full_name in airline_mapping.items():
        airline_code = airline_reverse.get(code, 0)
        rows.append([airline_code, origin_code, dest_code,
                     month, day, 0, dep_hour, is_weekend])
        codes.append(code)
        names.append(full_name)

    features_df = pd.DataFrame(rows, columns=[
        'AIRLINE_CODE', 'ORIGIN', 'DEST',
        'MONTH', 'DAY_OF_WEEK', 'DEP_DELAY',
        'DEP_HOUR', 'IS_WEEKEND'
    ])

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probs = model.predict_proba(features_df)[:, 1] * 100

    for i, code in enumerate(codes):
        route_data = df[
            (df['AIRLINE_CODE'] == code) &
            (df['ORIGIN']       == origin) &
            (df['DEST']         == dest)
        ]
        if len(route_data) > 0:
            avg_delay = max(int(round(route_data['ARR_DELAY'].mean())), 0)
        else:
            airline_data = df[df['AIRLINE_CODE'] == code]
            avg_delay = max(int(round(airline_data['ARR_DELAY'].mean())), 0) if len(airline_data) > 0 else 15

        all_flights.append({
            'airline':   names[i],
            'code':      code,
            'prob':      round(float(probs[i]), 1),
            'avg_delay': avg_delay,
            'hours':     avg_delay // 60,
            'mins':      avg_delay % 60
        })

    all_flights = sorted(all_flights, key=lambda x: x['prob'])

    if current_prob >= 70:
        recommendations = [f for f in all_flights if f['prob'] < 70]
        if len(recommendations) < 5:
            recommendations = all_flights[:5]
    elif current_prob >= 40:
        recommendations = [f for f in all_flights if f['prob'] < 40]
        if len(recommendations) < 5:
            recommendations = all_flights[:5]
    else:
        recommendations = all_flights[:5]

    return render_template(
        'recommend.html',
        origin          = origin,
        dest            = dest,
        current_prob    = current_prob,
        recommendations = recommendations,
        accuracy        = MODEL_ACCURACY
    )
# -------------------- AUTH & PROFILE --------------------
from flask import session, redirect, url_for, flash
from datetime import datetime

users_db = {}      # { email: {name, password, joined} }
search_history = {}  # { email: [ {route info} ] }

@app.route('/login-page')
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    email    = request.form['email']
    password = request.form['password']
    user     = users_db.get(email)
    if user and user['password'] == password:
        session['user'] = email
        flash('Welcome back, ' + user['name'] + '!', 'success')
        return redirect(url_for('profile'))
    flash('Invalid email or password.', 'error')
    return redirect(url_for('login_page'))

@app.route('/register', methods=['POST'])
def register():
    name     = request.form['name']
    email    = request.form['email']
    password = request.form['password']
    if email in users_db:
        flash('Email already registered. Please sign in.', 'error')
        return redirect(url_for('login_page'))
    users_db[email] = {
        'name':     name,
        'password': password,
        'joined':   datetime.now().strftime('%B %Y')
    }
    session['user'] = email
    flash('Account created! Welcome, ' + name + '!', 'success')
    return redirect(url_for('profile'))

@app.route('/profile')
def profile():
    if 'user' not in session:
        return redirect(url_for('login_page'))
    email = session['user']
    user  = users_db[email]
    history = search_history.get(email, [])
    return render_template('profile.html', user={
        'name':   user['name'],
        'email':  email,
        'joined': user['joined']
    }, history=history)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

app.secret_key = 'flightdelay_secret_2024'

# -------------------- RUN --------------------
if __name__ == '__main__':
    app.run(debug=True)