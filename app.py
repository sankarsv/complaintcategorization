from flask import Flask, render_template, request
import built as b

app = Flask(__name__)

@app.route("/getProduct", methods = ["GET", "POST"])
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

# @app.route("/sub", methods = ['POST'])
# def submit():
#     if request.method == "POST":
#         name = request.form["Issue"]
#     return render_template("sub.html", n = name)

if __name__ == "__main__":
    app.run(debug=True)