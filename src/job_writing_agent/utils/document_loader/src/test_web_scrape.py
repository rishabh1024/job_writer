from spider import Spider

app = Spider(api_key="sk-bcdc901b-5907-464d-a20b-4907275e1d63")
result = app.scrape_url(
    "https://proofpoint.wd5.myworkdayjobs.com/ProofpointCareers/job/Bengaluru-India---Remote/Senior-Software-Engineer_R13403?source=LinkedIn",
    params={"return_format": "markdown"},
)
print(result[0]["content"])
