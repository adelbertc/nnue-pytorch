"""
Library for invoking Stockfish-in-cog (https://github.com/replicate/cog)
hosted at a remote endpoint.
"""

import requests


class RemoteInference:
    def __init__(self, endpoint, token):
        session = requests.Session()
        default_headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer {}".format(token),
        }
        session.headers.update(default_headers)
        self.session = session
        self.endpoint = endpoint

        self.headers = default_headers

    def evaluate(self, fen):
        payload = {"input": {"fen": fen}}

        try:
            response = self.session.post(self.endpoint, json=payload)
            response = response.json()
            evaluation = response["output"]
            return evaluation
        except:
            print("Error: {}".format(response))
