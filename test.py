if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from model import Encoder, Decoder, Model, test_loader

    batch_size = 100
    x_dim = 784
    hidden_dim = 400
    latent_dim = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = Encoder(input_dim=784, hidden_dim=400, latent_dim=200)
    decoder = Decoder(latent_dim=200, hidden_dim=400, output_dim=784)

    model = Model(Encoder=encoder, Decoder=decoder).to(device)

    model.load_state_dict(torch.load("vae.pth", map_location=device))
    model.eval()


    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(test_loader)):
            x = x.view(batch_size, x_dim)
            x = x.to(device)
            x_hat, _, _ = model(x)
            break

    def show_image(x, idx):
        x = x.view(batch_size, 28, 28)
        plt.imshow(x[idx].cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.show()


    # show original vs reconstructed
    show_image(x, 0)
    show_image(x_hat, 0)