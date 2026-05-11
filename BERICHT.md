# 📊 Crypto Signal Bot - Projektbericht

## Übersicht

Der **Crypto Signal Bot** ist ein Echtzeit-Signalgenerator für Binance USDT-M Perpetual Futures. Er analysiert Kryptowährungen und generiert LONG/SHORT Trading-Signale mit präzisen Entry-, Stop-Loss- und Take-Profit-Levels.

> **Wichtig:** Dies ist ein reiner Signal-Bot. Er führt KEINE automatischen Trades aus.

---

## 🎯 Was macht der Bot?

| Funktion | Beschreibung |
|----------|--------------|
| **Marktdaten** | Holt Echtzeitdaten von Binance (OHLCV, Ticker) |
| **Technische Analyse** | Berechnet Indikatoren, Zonen, Muster |
| **Timing-Analyse** | Gann, Square of 9, Zyklen, Mondphasen, FOMC |
| **Signale** | Generiert LONG/SHORT/EXIT mit Entry/SL/TP |
| **Benachrichtigung** | Sendet Signale per Telegram oder Konsole |

---

## 📁 Projektstruktur

```
crypto_signal_bot/
├── main.py              # Hauptprogramm
├── config.yaml          # Konfiguration
├── requirements.txt     # Abhängigkeiten
├── README.md            # Dokumentation
├── bot/                 # 13 Module
│   ├── exchange.py      # Binance-Verbindung
│   ├── indicators.py    # SMA, EMA, RSI, ATR
│   ├── zones.py         # Support/Resistance
│   ├── patterns.py      # Fraktale, Kerzenmaster
│   ├── gann.py          # Gann-Winkel, Square of 9
│   ├── time_cycles.py   # 52-Zyklen, Mondphasen
│   ├── calendar_events.py # FOMC-Kalender
│   ├── signals.py       # Scoring-Engine
│   ├── risk.py          # Risikomanagement
│   ├── notifier.py      # Telegram
│   ├── storage.py       # Cooldown-Tracking
│   └── utils.py         # Hilfsfunktionen
└── tests/               # 65 Unit-Tests
```

---

## 🧮 Scoring-System

### Technischer Score (max. ~11 Punkte)
| Faktor | Punkte |
|--------|--------|
| Trend-Übereinstimmung | +2 |
| In S/R-Zone | +2 |
| Fraktal-Bestätigung | +1 |
| Kerzenmuster | +2 |
| Volumen-Spike | +2 |
| Wave 3 Setup | +2 |
| RSI-Divergenz (gegen) | -2 |

### Timing Score (max. 4 Punkte)
| Faktor | Punkte |
|--------|--------|
| Gann-Winkel-Konfluenz | 0-2 |
| Square of 9 Nähe | 0-2 |
| 52-Zyklus-Fenster | 0-1 |
| Mond-Ereignisfenster | 0-1 |
| FOMC Volatilitätsfenster | -1 bis 0 |

**Signal-Schwelle:** 7 (konfigurierbar)

---

## 📱 Beispiel-Signal

```
🔴 SHORT Signal: BTC/USDT:USDT

📊 Score: 8.0/7 (Tech: 6, Timing: 2.0)
📈 Trend: DOWN
💰 Price: 73,322.30

━━━━ Levels ━━━━
▫️ Entry: 73,322.30
🛑 Stop Loss: 74,208.75 (1.21%)
🎯 TP1: 72,435.85 (1:1)
🎯 TP2: 71,549.41 (1:2)

━━━━ Technische Analyse ━━━━
  ✓ Trend: down
  ○ Near support zone
  ✓ Fractal high @ 74,097.60
  ✓ Wave 3 setup

━━━━ Timing ━━━━
  Sq9: 73,254.62 (0.09%, score: 2.0)
```

---

## 🚀 Verwendung

### Einmal-Scan (Test)
```bash
docker-compose run --rm signal-bot python main.py --once
```

### Kontinuierlicher Betrieb
```bash
docker-compose run --rm signal-bot python main.py
```

### Mit Telegram
```bash
export TELEGRAM_BOT_TOKEN="dein_token"
export TELEGRAM_CHAT_ID="deine_chat_id"
docker-compose run --rm signal-bot python main.py
```

---

## ⚙️ Konfiguration

In `config.yaml` anpassbar:
- **Symbole:** BTC, ETH, SOL (erweiterbar)
- **Timeframes:** 4h (Trend), 15m (Entry), 1h (Zonen)
- **Scan-Intervall:** 300 Sekunden (5 Min)
- **Schwellenwert:** 7 Punkte
- **Risiko:** 1:1 und 1:2 R:R für TP1/TP2

---

## 📈 Technologien

- **Python 3.11**
- **ccxt** - Exchange API
- **pandas/numpy** - Datenanalyse
- **ephem** - Mondphasen
- **Docker** - Containerisierung
- **pytest** - 63/65 Tests bestanden

---

## 📅 Erstellt

**Datum:** 4. Februar 2026  
**Version:** 1.0.0

---

*Dieser Bot generiert nur Signale. Trading-Entscheidungen und Ausführung liegen beim Benutzer.*
