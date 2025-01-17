import torch


def main():
    print("Hello from gpt-qwen-distillation!")
    print(f"{torch.cuda.is_available()=}")

    x = torch.rand(5, 3)
    print(f"{x=}")


if __name__ == "__main__":
    main()
