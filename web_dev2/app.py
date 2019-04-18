from __future__ import division
import pandas as pd
from flask import Flask, render_template, request, jsonify
import pickle
import json

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')




@app.route('/input_data', methods=['GET', 'POST'])
def input_data():
    data = json.loads(request.data)

    def row_info(x):
        df = pd.read_excel("website_display.xlsx")
        text = "<p>Tweet Post: " + df['text'][x] + "</p>" + "\n\n" + "<p>Description: " + str(df['description'][x]) + "</p>"

        stats = "<p>\n" + "Followers: " + str(df['followers_count'][x]) + "\n" + "Favourites: " + str(df['favourites_count'][x]) + "\n" + "Statuses: " + str(df['statuses_count'][x]) + "\n" + "</p>"

        prob1 = "<p>\n"+ "probability of bot generated: " + "{percent:.2%}".format(percent=df['bot_prob'][x]) + "</p>"
        prob2 = "<p>\n"+"probability of corporate posted: " + "{percent:.2%}".format(percent=df['corporate_prob'][x]) + "</p>"
        prob3 = "<p>\n"+"probability of hijacking keywords: " + "{percent:.2%}".format(percent=df['hijack_prob'][x]) + "</p>"
        prob4 = "<p>\n"+"probability of marketing spam: " + "{percent:.2%}".format(percent=df['marketing_prob'][x]) + "</p>"
        return str(text+"\n"+stats+"\n"+prob1+"\n"+prob2+"\n"+prob3+"\n"+prob4)

    output = row_info(int(data['result']))
    return jsonify({'result': output})




if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
