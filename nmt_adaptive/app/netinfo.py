"""Адреса комп'ютера вчителя в шкільній мережі — для режиму класу (без streamlit).

Запуск:  python -m app.netinfo   → друкує адресу, яку учні відкривають у браузері.
"""
import socket

import config


def local_ip() -> str:
    """IP цього комп'ютера в локальній мережі (напр. 192.168.1.10) або «localhost», якщо мережі немає.

    UDP-«з'єднання» нічого не надсилає: система лише обирає мережеву карту, через яку пішов би пакет.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.255.255.255", 1))
            return sock.getsockname()[0]
    except OSError:
        return "localhost"


def class_url(port: int = config.CLASS_PORT) -> str:
    """Адреса тесту для учнів: http://<IP>:<порт>."""
    return f"http://{local_ip()}:{port}"


if __name__ == "__main__":
    print(f"Учні відкривають у браузері:  {class_url()}")
    print(f"Сторінка вчителя (лише тут):  http://localhost:{config.TEACHER_PORT}")
