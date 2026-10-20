from locust import HttpUser, between, task


class InferenceUser(HttpUser):
    wait_time = between(0.0, 0.1)

    @task(4)
    def text(self) -> None:
        self.client.post("/predict_text", json={"text": "python sql pipeline deploy cloud"})

    @task(1)
    def health(self) -> None:
        self.client.get("/health")
