import os
import torch
import torch.nn as nn
from torch.utils.data import Subset
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from Generator import Generator
from Discriminator import Discriminator
from config import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUT_DIR, exist_ok=True)

def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    # train_dataset = torch.utils.data.Subset(train_dataset, range(1000)) 
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, test_loader

def compute_real_width(image):
    b = image.size(0)
    img = (image.squeeze(0).cpu().numpy() > 0).reshape(b, -1).astype(float) 
    temp = img.sum(axis=1) / img.shape[1]
    _max = np.max(temp)
    _min = np.min(temp)
    return (temp - _min) / (abs(_max - _min) + 1e-8)

def train_one_epoch(netG, netD, optimizerG, optimizerD, train_loader, epoch):
    adversarial_loss = nn.BCEWithLogitsLoss()
    auxiliary_loss = nn.CrossEntropyLoss()
    width_loss = nn.MSELoss()

    real_label, fake_label = 1., 0.
    loop = tqdm(train_loader, desc=f"Эпоха {epoch}/{NUM_EPOCHS}")

    total_d_loss, total_g_loss = 0, 0
    for i, (real_images, labels) in enumerate(loop):
        real_images, labels = real_images.to(DEVICE), labels.to(DEVICE)
        b_size = real_images.size(0)

        netD.zero_grad()
        real_validity, real_aux, real_width_pred = netD(real_images)
        valid = torch.full((b_size,), real_label, device=DEVICE)
        d_real_loss = adversarial_loss(real_validity, valid)
        d_class_loss = auxiliary_loss(real_aux, labels)
        real_width_targets = torch.tensor(compute_real_width(real_images), device=DEVICE, dtype=torch.float)
        d_real_width_loss = width_loss(real_width_pred, real_width_targets)

        noise = torch.randn(b_size, NZ, device=DEVICE)
        gen_labels = torch.randint(0, NUM_CLASSES, (b_size,), device=DEVICE)
        widths = torch.rand(b_size, device=DEVICE) * 2.0
        fake_images = netG(noise, gen_labels, widths)
        fake_validity, fake_aux, fake_width_pred = netD(fake_images.detach())
        fake = torch.full((b_size,), fake_label, device=DEVICE)
        d_fake_loss = adversarial_loss(fake_validity, fake)
        d_fake_width_loss = width_loss(fake_width_pred, widths)

        d_loss = d_real_loss + d_fake_loss + d_class_loss + d_real_width_loss + d_fake_width_loss
        d_loss.backward()
        optimizerD.step()
        
        netG.zero_grad()
        fake_validity, fake_aux, fake_width_pred = netD(fake_images)
        valid = torch.full((b_size,), real_label, device=DEVICE)
        g_adv_loss = adversarial_loss(fake_validity, valid)
        g_class_loss = auxiliary_loss(fake_aux, gen_labels)
        g_width_loss = width_loss(fake_width_pred, widths)
        g_loss = g_adv_loss + g_class_loss + g_width_loss
        g_loss.backward()
        optimizerG.step()

        total_d_loss += d_loss.item()
        total_g_loss += g_loss.item()

        if i % PRINT_EVERY == 0:
            loop.set_postfix({
                'Ошибка D': f'{d_loss.item():.4f}',
                'Ошибка G': f'{g_loss.item():.4f}'
            })
    return total_d_loss / len(train_loader), total_g_loss / len(train_loader)

def evaluate(netD, test_loader, netG):
    adversarial_loss = nn.BCEWithLogitsLoss()
    auxiliary_loss = nn.CrossEntropyLoss()
    width_loss = nn.MSELoss()

    netD.eval()
    total_real_adv, total_real_class, total_real_width = 0, 0, 0
    total_fake_adv, total_fake_class, total_fake_width = 0, 0, 0

    correct_real_disc, correct_fake_disc = 0, 0
    correct_real_class, correct_fake_class = 0, 0
    total_real, total_fake = 0, 0

    with torch.no_grad():
        for real_images, labels in test_loader:
            real_images, labels = real_images.to(DEVICE), labels.to(DEVICE)
            b_size = real_images.size(0)

            # === Реальные ===
            real_validity, real_aux, real_width_pred = netD(real_images)
            valid = torch.ones(b_size, device=DEVICE)
            real_width_targets = torch.tensor(compute_real_width(real_images),
                                              device=DEVICE, dtype=torch.float)

            d_real_loss = adversarial_loss(real_validity, valid)
            d_class_loss = auxiliary_loss(real_aux, labels)
            d_real_width_loss = width_loss(real_width_pred, real_width_targets)

            total_real_adv += d_real_loss.item()
            total_real_class += d_class_loss.item()
            total_real_width += d_real_width_loss.item()

            preds_disc = (torch.sigmoid(real_validity) > 0.5).long()
            correct_real_disc += preds_disc.eq(valid.long()).sum().item()
            preds_class = real_aux.argmax(dim=1)
            correct_real_class += preds_class.eq(labels).sum().item()
            total_real += b_size

            # === Фейковые ===
            noise = torch.randn(b_size, NZ, device=DEVICE)
            gen_labels = torch.randint(0, NUM_CLASSES, (b_size,), device=DEVICE)
            widths = torch.rand(b_size, device=DEVICE) * 2.0
            fake_images = netG(noise, gen_labels, widths)

            fake_validity, fake_aux, fake_width_pred = netD(fake_images)
            fake = torch.zeros(b_size, device=DEVICE)

            d_fake_loss = adversarial_loss(fake_validity, fake)
            d_fake_class_loss = auxiliary_loss(fake_aux, gen_labels)
            d_fake_width_loss = width_loss(fake_width_pred, widths)

            total_fake_adv += d_fake_loss.item()
            total_fake_class += d_fake_class_loss.item()
            total_fake_width += d_fake_width_loss.item()

            preds_disc = (torch.sigmoid(fake_validity) > 0.5).long()
            correct_fake_disc += preds_disc.eq(fake.long()).sum().item()
            preds_class = fake_aux.argmax(dim=1)
            correct_fake_class += preds_class.eq(gen_labels).sum().item()
            total_fake += b_size

    # === усреднение ===
    avg_real_adv = total_real_adv / len(test_loader)
    avg_real_class = total_real_class / len(test_loader)
    avg_real_width = total_real_width / len(test_loader)
    avg_fake_adv = total_fake_adv / len(test_loader)
    avg_fake_class = total_fake_class / len(test_loader)
    avg_fake_width = total_fake_width / len(test_loader)

    # === суммарные ошибки ===
    avg_d_loss = avg_real_adv + avg_real_class + avg_real_width + \
                 avg_fake_adv + avg_fake_class + avg_fake_width
    avg_g_loss = avg_fake_adv + avg_fake_class + avg_fake_width

    # === точности ===
    real_disc_acc = 100.0 * correct_real_disc / total_real if total_real > 0 else 0
    real_class_acc = 100.0 * correct_real_class / total_real if total_real > 0 else 0
    fake_disc_acc = 100.0 * correct_fake_disc / total_fake if total_fake > 0 else 0
    fake_class_acc = 100.0 * correct_fake_class / total_fake if total_fake > 0 else 0

    return {
        "d_loss": avg_d_loss,
        "g_loss": avg_g_loss,
        "real_disc_acc": real_disc_acc,
        "fake_disc_acc": fake_disc_acc,
        "real_class_acc": real_class_acc,
        "fake_class_acc": fake_class_acc
    }

def plot_losses(history, epoch):
    plt.figure(figsize=(12, 8))

    # Ошибка генератора
    plt.subplot(2, 2, 1)
    plt.plot(history["epoch"], history["g_loss"], label="G (train)")
    plt.plot(history["epoch"], history["test_g_loss"], label="G (test)")
    plt.xlabel("Эпоха")
    plt.ylabel("Ошибка")
    plt.legend()
    plt.title("Ошибка генератора")

    # Ошибка дискриминатора
    plt.subplot(2, 2, 2)
    plt.plot(history["epoch"], history["d_loss"], label="D (train)")
    plt.plot(history["epoch"], history["test_d_loss"], label="D (test)")
    plt.xlabel("Эпоха")
    plt.ylabel("Ошибка")
    plt.legend()
    plt.title("Ошибка дискриминатора")

    # Точность дискриминации
    plt.subplot(2, 2, 3)
    plt.plot(history["epoch"], history["real_disc_acc"], label="D acc (реальные)")
    plt.plot(history["epoch"], history["fake_disc_acc"], label="D acc (фейковые)")
    plt.xlabel("Эпоха")
    plt.ylabel("Точность (%)")
    plt.legend()
    plt.title("Точность дискриминации")

    # Точность классификации
    plt.subplot(2, 2, 4)
    plt.plot(history["epoch"], history["real_class_acc"], label="Class acc (реальные)")
    plt.plot(history["epoch"], history["fake_class_acc"], label="Class acc (фейковые)")
    plt.xlabel("Эпоха")
    plt.ylabel("Точность (%)")
    plt.legend()
    plt.title("Точность классификации")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"losses_{epoch}.png"))
    plt.close()
    print("Графики ошибок и метрик сохранены!")

def sample_images(netG, epoch, fixed_noise, fixed_labels, fixed_widths):
    netG.eval()
    with torch.no_grad():
        fake = netG(fixed_noise, fixed_labels, fixed_widths).detach().cpu()
    grid = utils.make_grid(fake, nrow=10, normalize=True, scale_each=True)
    utils.save_image(grid, os.path.join(OUT_DIR, f'epoch_{epoch:03d}.png'))
    print(f"Сохранены сгенерированные изображения для эпохи {epoch}")

def save_checkpoint(netG, netD, epoch):
    torch.save({
        'epoch': epoch,
        'netG_state': netG.state_dict(),
        'netD_state': netD.state_dict(),
    }, os.path.join(OUT_DIR, f'checkpoint_epoch_{epoch:03d}.pt'))
    print(f"Чекпоинт сохранён: эпоха {epoch}")

def train():
    print(f"Используется устройство: {DEVICE}")
    train_loader, test_loader = get_dataloaders()
    netG, netD = Generator().to(DEVICE), Discriminator().to(DEVICE)
    # checkpoint = torch.load(os.path.join(OUT_DIR, f'checkpoint_epoch_010.pt'), map_location=DEVICE)

    # netG.load_state_dict(checkpoint['netG_state'])
    # netD.load_state_dict(checkpoint['netD_state'])

    optimizerD = optim.AdamW(netD.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizerG = optim.AdamW(netG.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    fixed_noise = torch.randn(10 * 10, NZ, device=DEVICE)
    fixed_labels = torch.arange(0, 10, dtype=torch.long, device=DEVICE).repeat(10)
    fixed_widths = torch.linspace(0., 3, steps=10, device=DEVICE).repeat_interleave(10)

    print("Предварительная генерация до обучения...")
    sample_images(netG, 0, fixed_noise, fixed_labels, fixed_widths)

    history = {
        "epoch": [],
        "d_loss": [], "g_loss": [],
        "test_d_loss": [], "test_g_loss": [],
        "real_disc_acc": [], "fake_disc_acc": [],
        "real_class_acc": [], "fake_class_acc": []
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        history["epoch"].append(epoch)
        d_loss, g_loss = train_one_epoch(netG, netD, optimizerG, optimizerD, train_loader, epoch)

        history["d_loss"].append(d_loss)
        history["g_loss"].append(g_loss)

        results = evaluate(netD, test_loader, netG)

        history["test_d_loss"].append(results["d_loss"])
        history["test_g_loss"].append(results["g_loss"])
        history["real_disc_acc"].append(results["real_disc_acc"])
        history["fake_disc_acc"].append(results["fake_disc_acc"])
        history["real_class_acc"].append(results["real_class_acc"])
        history["fake_class_acc"].append(results["fake_class_acc"])

        if epoch % SAMPLE_EVERY_EPOCHS == 0 or epoch == NUM_EPOCHS:
            sample_images(netG, epoch, fixed_noise, fixed_labels, fixed_widths)
            # save_checkpoint(netG, netD, epoch)
            plot_losses(history, epoch)

    print("Обучение завершено.")

if __name__ == "__main__":
    train()
