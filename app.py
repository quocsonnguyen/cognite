import sys
sys.path.append('./code')

from typing_extensions import runtime
from flask import Flask, request, session, render_template
import pandas as pd
from suggest_new_experiment import get_frequency


app = Flask(__name__, template_folder='templates', static_folder='assets')
app.config['SECRET_KEY'] = "secret"

DATA_PATH = './data/'
DATA_FILENAME = 'data_copy.csv'

@app.route('/')
def index():
    df = pd.read_csv(DATA_PATH + DATA_FILENAME, header=None)
    df = df.drop(1, axis=1)
    table_data = df.values.tolist()
    new_table_data = []
    for row in table_data:
        new_table_data.append([round(row[0],1), round(row[1], 2), float("{:.1f}".format(row[2])), row[3]])

    return render_template('index.html', table_data=new_table_data)

@app.route('/api/load-table')
def load_table():
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
            'code' : 2,
            'msg' : 'Load table failed'
        }

@app.route('/api/get-freq')
def get_freq():
    try:
        ps = request.args.get('ps')
        ps = float(ps)
        
        freq = ps
        freq = get_frequency('./data/data.csv', ps)

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
        added_time = request.form['addedTime']

        with open(DATA_PATH + DATA_FILENAME, 'a') as f:
            f.write(f'{freq}, {col2}, {ps}, {p}, {added_time}')
            f.write("\n")
        
        return {
            'code' : 0,
            'msg' : 'Write new line successfully'
        }
    except:
        return {
            'code' : 2,
            'msg' : 'Write new line failed'
        }

@app.route('/api/delete-record', methods=['POST'])
def delete_record():
    try:
        row_id = int(request.form['rowId'])
        df = pd.read_csv(DATA_PATH + DATA_FILENAME, header=None)
        df = df.drop(row_id-1, axis=0)
        df.to_csv(DATA_PATH + DATA_FILENAME, header=False, index=False)

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
    app.run(debug=True)
