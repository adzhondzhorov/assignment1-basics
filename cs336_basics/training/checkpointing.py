import pickle


def save_checkpoint(model, optimizer, iteration, path):
    checkpoint = {
        "iteration": iteration,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    with open(path, "wb+") as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(path, model, optimizer):
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
        if model is not None:
            model.load_state_dict(checkpoint["model_state"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        iteration = checkpoint["iteration"]
        return iteration
