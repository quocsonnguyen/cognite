import os
import sys
sys.path.append('./code')

from flask import Flask, request, session, render_template, redirect
from flask_sqlalchemy import SQLAlchemy
from  werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, time
import pandas as pd
import uuid
from suggest_new_experiment import get_frequency
app = Flask(__name__, template_folder='templates', static_folder='assets')
app.config['SECRET_KEY'] = 'cognite secret 12/2021'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

db = SQLAlchemy(app)
DATA_PATH = './data/'

class History(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    public_id = db.Column(db.String(50), unique = True)
    email = db.Column(db.String(70))
    name = db.Column(db.String(100))
    time = db.Column(db.String(70))
    event = db.Column(db.String(80))

class User(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    public_id = db.Column(db.String(50), unique = True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(70), unique = True)
    phoneNumber = db.Column(db.String(15), unique = True)
    password = db.Column(db.String(80))
    role = db.Column(db.String(50))

def write_history(e):
    if session['user_role'] == 'user':
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        history = History(
            public_id = str(uuid.uuid4()),
            email = session['user_email'],
            name = session['username'],
            time = dt_string,
            event = e
        )
        # insert user
        db.session.add(history)
        db.session.commit()

def last_activity():
    user = User.query.all()
    table_last_activity = []
    for u in user:
        history = History.query\
                .filter_by(email = u.email)\
                .order_by(History.time.desc())\
                .first()             
        if history:
            table_last_activity.append(history)
    return table_last_activity

@app.route('/')
def index():
    if 'isLogged' not in session:
        return redirect('/login')

    try:
        DATA_FILENAME = session['user_filename']
        df = pd.read_csv(DATA_PATH + DATA_FILENAME, header=None)
        df = df.drop(1, axis=1)
        table_data = df.values.tolist()
        new_table_data = []
        for row in table_data:
            new_table_data.append([round(row[0],1), round(row[1], 2), float("{:.1f}".format(row[2])), row[3]])

        return render_template('index.html', username=session['username'], table_data=new_table_data)
    except:
        return render_template('index.html', username=session['username'])

@app.route('/admin')
def admin():
    if 'isLogged' not in session:
        return redirect('/login')
    elif session['user_role'] == "admin":
        try:
            history = History.query.all()
            df = pd.read_csv(DATA_PATH + 'GLOBAL.csv', header=None, names=range(6))
            df = df.drop(1, axis=1)
            table_data = df.values.tolist()
            new_table_data = []
            for row in table_data:
                new_table_data.append([round(row[0],1), round(row[1], 2), float("{:.1f}".format(row[2])), row[3]])
            last_activitys = last_activity()
            return render_template('admin.html', table_data=new_table_data, table_history=history, table_last_activity=last_activitys)
        except:
            return render_template('admin.html')
    else:
        return redirect('/')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'isLogged' in session:
        return redirect('/')
    if request.method == 'GET':
        return render_template('login.html')
    elif request.method == 'POST':
        auth = request.form
  
        if not auth or not auth.get('email') or not auth.get('password'):
            return render_template('login.html', message="Please fill both the username and password fields!")
    
        user = User.query\
            .filter_by(email = auth.get('email'))\
            .first()
    
        if not user:
            return render_template('login.html', message="User does not exist!")
    
        if check_password_hash(user.password, auth.get('password')):
            session['isLogged'] = True
            session['username'] = user.name
            session['user_filename'] = user.email+'.csv'
            session['user_email'] = user.email
            session['user_role'] = user.role
            write_history('login')
            if user.role == 'user':
                return redirect('/')
            else:
                return redirect('/admin')

        # password is wrong
        return render_template('login.html', message="Wrong password!")

@app.route('/logout')
def logout():
    write_history('logout')
    session.clear()
    return redirect('/login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    elif request.method == 'POST':
        data = request.form
        name, email, phoneNumber = data.get('name'), data.get('email'), data.get('phoneNumber')
        password, re_password = data.get('password'), data.get('re-password')

        if password != re_password:
            return render_template('/register.html', message='Confirm password does not match password!')

        if len(password) < 8:
            return render_template('/register.html', message='Password must have at least 8 letters!')

        user = User.query\
            .filter_by(email = email)\
            .first()
        if not user:
            # database ORM object
            try:
                user = User(
                public_id = str(uuid.uuid4()),
                name = name,
                email = email,
                phoneNumber = phoneNumber,
                password = generate_password_hash(password),
                role = "user"
                )
                # insert user
                db.session.add(user)
                db.session.commit()

                # create a new csv file for this user
                open(DATA_PATH + email + '.csv', 'x')
                open(DATA_PATH + email + '_BACKUP.csv', 'x')
                return render_template('registerSuccess.html')
            except:
                return render_template('/register.html', message='This phone number is already in use')
        else:
            return render_template('/register.html', message='User already exists. Please Log in.')

@app.route('/api/load-table')
def load_table():
    try:
        DATA_FILENAME = session['user_filename']
        try:
            df = pd.read_csv(DATA_PATH + DATA_FILENAME, header=None)
            df = df.drop(1, axis=1)
            table_data = df.values.tolist()
            new_table_data = []
            for row in table_data:
                new_table_data.append([round(row[0],1), round(row[1], 2), float("{:.1f}".format(row[2])), row[3]])
            return {
                'code' : 0,
                'data' : new_table_data
            }
        except:
            return {
                'code' : 1,
                'msg' : 'Table is empty'
            }
    except:
        return {
            'code' : 2,
            'msg' : 'Load table failed'
        }

@app.route('/api/load-global-table')
def load_global_table():
    try:
        try:
            df = pd.read_csv(DATA_PATH + 'GLOBAL.csv', header=None, names=range(6))
            df = df.drop(1, axis=1)
            table_data = df.values.tolist()
            new_table_data = []
            for row in table_data:
                new_table_data.append([round(row[0],1), round(row[1], 2), float("{:.1f}".format(row[2])), row[3]])
            return {
                'code' : 0,
                'data' : new_table_data
            }
        except:
            return {
                'code' : 1,
                'msg' : 'Table is empty'
            }
    except:
        return {
            'code' : 2,
            'msg' : 'Load table failed'
        }

@app.route('/api/get-freq')
def get_freq():
    try:
        ps = request.args.get('ps')
        ps = float(ps)
        
        freq = get_frequency(DATA_PATH + 'GLOBAL.csv', ps)
        write_history('get freq')

        return {
            'code' : 0,
            'freq' : freq,
        }
    except:
        return {
            'code' : 2,
            'msg' : 'Can not get frequency'
        }

@app.route('/api/save-new-record', methods=['POST'])
def save_new_record():
    try:
        freq = request.form['freq']
        col2 = 0
        ps = request.form['ps']
        p = request.form['p']
        now = datetime.now()
        added_time = now.strftime("%d/%m/%Y %H:%M:%S")

        user_email = session['user_email']
        DATA_FILENAME = session['user_filename']
        DATA_FILENAME_BACKUP = DATA_FILENAME.replace('.csv', '_BACKUP.csv')

        with open(DATA_PATH + DATA_FILENAME, 'a') as f:
            f.write(f'{freq}, {col2}, {ps}, {p}, {added_time}')
            f.write("\n")

        with open(DATA_PATH + DATA_FILENAME_BACKUP, 'a') as f:
            f.write(f'{freq}, {col2}, {ps}, {p}, {added_time}')
            f.write("\n")

        with open(DATA_PATH + 'GLOBAL.csv', 'a') as f:
            f.write(f'{freq}, {col2}, {ps}, {p}, {added_time}, {user_email}')
            f.write("\n")

        with open(DATA_PATH + 'GLOBAL_BACKUP.csv', 'a') as f:
            f.write(f'{freq}, {col2}, {ps}, {p}, {added_time}, {user_email}')
            f.write("\n")
        
        write_history('save new record')
        
        return {
            'code' : 0,
            'msg' : 'Write new line successfully',
            'addedTime' : added_time
        }
    except:
        return {
            'code' : 2,
            'msg' : 'Write new line failed'
        }

@app.route('/api/delete-record', methods=['POST'])
def delete_record():
    if 'isLogged' not in session:
        return {
            'code' : 5,
            'msg' : 'You do not have permission'
        }

    try:
        DATA_FILENAME = session['user_filename']
        row_id = int(request.form['rowId'])
        df = pd.read_csv(DATA_PATH + DATA_FILENAME, header=None)
        row_data = df.iloc[row_id-1]
        df = df.drop(row_id-1, axis=0)
        df.to_csv(DATA_PATH + DATA_FILENAME, header=False, index=False)

        # need to delete from GLOBAL
        user_email = session['user_email']
        time_added = row_data.iloc[4]

        with open(DATA_PATH + "GLOBAL.csv", "r+") as f:
            lines = f.readlines()
            f.seek(0)
            for line in lines:
                tokens = line.split(',')
                try:
                    if tokens[4] != time_added and tokens[5] != user_email:
                        f.write(line)
                except:
                    f.write(line)
            f.truncate()

        write_history('delete record')

        return {
            'code' : 0,
            'msg' : 'deleted'
        }
    except:
        return {
            'code' : 2,
            'msg' : 'Can not delete'
        }

@app.route('/api/delete-record-in-global', methods=['POST'])
def delete_record_in_global():
    if 'isLogged' not in session:
        return {
            'code' : 5,
            'msg' : 'You do not have permission'
        }

    try:
        row_id = int(request.form['rowId'])
        df = pd.read_csv(DATA_PATH + 'GLOBAL.csv', header=None, names=range(6))
        df = df.drop(row_id-1, axis=0)
        df.to_csv(DATA_PATH + 'GLOBAL.csv', header=False, index=False)

        return {
            'code' : 0,
            'msg' : 'deleted'
        }
    except:
        return {
            'code' : 2,
            'msg' : 'Can not delete'
        }

if __name__ == '__main__':
    app.run(debug=False)
