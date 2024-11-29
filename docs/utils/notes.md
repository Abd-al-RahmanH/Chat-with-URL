### Deploy on Hugging Face

3. **Log in to Hugging Face CLI:**

```sh
huggingface-cli login
```

2. **Create a new repository on Hugging Face.**

3. **Push the Docker image to Hugging Face:**

```sh
docker tag watsonx-webchat huggingface.co/ruslanmv/watsonx-webchat

```



```sh
docker push huggingface.co/ruslanmv/watsonx-webchat
```

4. **Configure the Hugging Face repository to use the Docker image:**

    - Go to your Hugging Face repository page.
    - Click on "Settings".
    - Under "Custom Docker Image", set the image to `huggingface.co/ruslanmv/watsonx-webchat`.