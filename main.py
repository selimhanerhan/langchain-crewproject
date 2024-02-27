from flask import Flask, request, render_template, send_file
from weasyprint import HTML

from agent import YoutubeChannelManager

import os

class Controller:
    def __init__(self):
        self.app = Flask(__name__, template_folder="web")

        @self.app.route("/", methods=["GET"])
        def read_root():
            return render_template("index.html")

        @self.app.route("/postInformation/", methods=["POST"])
        def post_information():
            keyword = request.form.get("user_input_keyword")
            url = request.form.get("user_input_url")

            manager = YoutubeChannelManager()
            result = manager.run_crew(keyword, url)

            return render_template("post_information.html", result=result)

        @self.app.route("/download/", methods=["POST"])
        def download():
            result = request.form.get("result")
            pdf = HTML(string=result).write_pdf()

            filename = "result.pdf"
            with open(filename, "wb") as file:
                file.write(pdf)

            return send_file(filename, as_attachment=True)

    def run(self):
        self.app.run(host="127.0.0.1", port=8000, debug=True)

# Usage
if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = ""

    controller = Controller()
    controller.run()
