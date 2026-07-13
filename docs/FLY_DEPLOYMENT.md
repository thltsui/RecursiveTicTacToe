# Fly.io Deployment Guide

This guide contains the step-by-step instructions to get the Ultimate Tic-Tac-Toe application live on the internet using Fly.io, complete with a Redis backend for horizontal scaling.

## Prerequisites

1. **Create an Account:** Sign up at [fly.io](https://fly.io/). You will need to enter a credit card for spam prevention, but this deployment should fall within their free tier.
2. **Install the CLI Tool:** Open your Mac terminal and run:
   ```bash
   brew install flyctl
   ```
3. **Login:** Link your terminal to your account by running:
   ```bash
   fly auth login
   ```

## Step-by-Step Deployment

Run all the following commands from the root directory of this project (`/Users/holungtsui/Documents/GitHub/UlltimateTicTacToe`).

### 1. Initial Launch
Since there is already a `fly.toml` and `Dockerfile` in the repository, you can launch the app.
```bash
fly launch
```
*Note: When asked if you want to copy the existing configuration, say **Yes**.*
*Note: When asked if you want to deploy now, you can say **No** because we need to set up Redis first.*

### 2. Set Up Redis (For Multiplayer Sync)
We need a Redis database so that if the app scales to multiple servers, the WebSocket rooms and game state remain synchronized. Fly partners with Upstash to provide this seamlessly.

```bash
fly redis create
```
- It will prompt you for a name (e.g., `ultimatettt-redis`).
- Select a region (preferably the same one you deployed your app in, e.g., `lhr`).
- Choose the free tier.

Once it's created, you need to attach it to your app so the code can access it via the `REDIS_URL` environment variable:
```bash
fly redis attach <your-redis-name> -a <your-app-name>
```

### 3. Deploy the Application
Now that Redis is hooked up, you can deploy the actual code!
```bash
fly deploy
```

### 4. Scale Up (Optional)
If you want to ensure the app is highly available and test the Redis synchronization, you can tell Fly to run multiple instances of your server:
```bash
fly scale count 2
```

## Success!
Your app will now be live at `https://<your-app-name>.fly.dev`. You can share this link with anyone, and they can play online immediately.

### Custom Domains (Optional)
If you purchase a custom domain later, you can link it to Fly.io by running `fly certs add yourdomain.com` and following their DNS instructions.
