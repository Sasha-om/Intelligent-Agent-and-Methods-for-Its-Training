"""Тести адреси для режиму класу (app/netinfo.py)."""
import socket

import config
from app import netinfo


def test_local_ip_is_address() -> None:
    """IP — чотири числа через крапку або «localhost»."""
    ip = netinfo.local_ip()
    assert ip == "localhost" or (len(ip.split(".")) == 4 and all(p.isdigit() for p in ip.split(".")))


def test_no_network_gives_localhost(monkeypatch) -> None:
    """Без мережі — «localhost», а не помилка."""
    class Broken:
        def __init__(self, *args) -> None:
            pass

        def __enter__(self):
            raise OSError("немає мережі")

        def __exit__(self, *args) -> None:
            pass
    monkeypatch.setattr(socket, "socket", Broken)
    assert netinfo.local_ip() == "localhost"


def test_class_url_uses_port(monkeypatch) -> None:
    """Адреса: http://<IP>:<CLASS_PORT>."""
    monkeypatch.setattr(netinfo, "local_ip", lambda: "192.168.1.10")
    assert netinfo.class_url() == f"http://192.168.1.10:{config.CLASS_PORT}"
