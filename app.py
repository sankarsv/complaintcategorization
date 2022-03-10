from flask import Flask, render_template, request
import built as b

app = Flask(__name__)

@app.route("/getProduct", methods = ["POST"])
def hello():

    if request.method == "POST":
        Issue = request.form['issue']
        print(Issue)
        str = b.prediction(Issue)
        print(str)
        return str.tostring()

    return render_template("index.html")

@app.route("/", methods = ["GET"])
def displayIndex():

    return render_template("complaintform.html")


if __name__ == "__main__":
    app.run(debug=True)