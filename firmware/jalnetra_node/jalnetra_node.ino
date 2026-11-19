/*
 * JalNetra field node firmware - ESP32 + SX1276 (raw LoRa, IN865)
 * -----------------------------------------------------------------
 * Wake every 15 min -> power sensors -> sample pH, ORP, turbidity, TDS, temperature,
 * flow -> pack 17-byte payload -> transmit -> deep sleep.
 *
 * The payload layout is IDENTICAL to jalnetra/codec.py (the simulator decodes the
 * same bytes), so the cloud side can be tested before any hardware exists.
 *
 * Libraries (Arduino Library Manager):
 *   - LoRa                     by Sandeep Mistry
 *   - Adafruit ADS1X15
 *   - OneWire, DallasTemperature
 * For a production LoRaWAN deployment swap the LoRa calls for MCCI LMIC / RadioLib
 * LoRaWAN (OTAA, IN865 channel plan); the 17-byte FRMPayload stays the same.
 *
 * Wiring (example):
 *   SX1276: SCK 18, MISO 19, MOSI 23, NSS 5, RST 14, DIO0 26
 *   ADS1115 (I2C 21/22): A0 pH, A1 ORP, A2 turbidity, A3 TDS
 *   DS18B20 data: GPIO 4 (4.7k pull-up)
 *   Flow sensor pulse: GPIO 27 (hall, open collector, pull-up)
 *   Sensor power MOSFET gate: GPIO 25 (HIGH = sensors on)
 *   Battery divider (100k/100k): GPIO 35
 *
 * Calibration constants below MUST be replaced with values from on-site calibration
 * (pH 4/7 buffers, ORP 225 mV / 475 mV standard, formazin turbidity standards,
 * 1413 uS/cm conductivity standard, bucket-and-stopwatch for the flow meter).
 */
#include <SPI.h>
#include <Wire.h>
#include <LoRa.h>
#include <Adafruit_ADS1X15.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// ------------------------------------------------------------------ config
#define NODE_ID          3          // 1..255, unique per node (N03 = 3)
#define SLEEP_MINUTES    15
#define LORA_FREQ        865.0625E6 // IN865 band
#define LORA_SF          9          // set per site from the link budget (sim: lora.py)
#define LORA_TX_DBM      14

#define PIN_LORA_NSS 5
#define PIN_LORA_RST 14
#define PIN_LORA_DIO0 26
#define PIN_ONEWIRE 4
#define PIN_FLOW 27
#define PIN_SENSOR_PWR 25
#define PIN_VBAT 35

#define FLOW_WINDOW_MS   30000UL    // count pulses for 30 s each wake
#define FLOW_K_PULSE_PER_L  (1.0f / 0.0083f)  // YF-DN25 ~ 120 pulses / L (calibrate!)
#define WARMUP_MS        8000UL     // probe settling after power-on

// pH: two-point calibration (volts at pH 7 and slope V/pH)
const float PH_V7 = 1.500f, PH_SLOPE = -0.0591f * 3.0f;   // amplified board, adjust
// ORP: board outputs (2.0 V - ORP_mV/1000); offset from 225 mV standard
const float ORP_OFFSET_MV = 0.0f;
// Turbidity: NTU = a*V^2 + b*V + c  (fit to formazin standards)
const float TURB_A = -1120.4f, TURB_B = 5742.3f, TURB_C = -4352.9f;
// TDS: cubic from Gravity TDS board, temperature compensated
const float TDS_K = 1.0f;

#define FLAG_LOW_BATT 0x01
#define FLAG_SELFTEST 0x02
#define FLAG_BOOT     0x04

RTC_DATA_ATTR uint16_t seq = 0;
RTC_DATA_ATTR uint32_t bootCount = 0;

Adafruit_ADS1115 ads;
OneWire oneWire(PIN_ONEWIRE);
DallasTemperature tempSensor(&oneWire);
volatile uint32_t pulses = 0;

void IRAM_ATTR onPulse() { pulses++; }

static float readVolts(uint8_t ch, uint8_t n = 16) {
  float acc = 0;
  for (uint8_t i = 0; i < n; i++) { acc += ads.computeVolts(ads.readADC_SingleEnded(ch)); delay(20); }
  return acc / n;
}

static void put16(uint8_t *b, int i, uint16_t v) { b[i] = v >> 8; b[i + 1] = v & 0xFF; }

void setup() {
  bootCount++;
  pinMode(PIN_SENSOR_PWR, OUTPUT);
  digitalWrite(PIN_SENSOR_PWR, HIGH);           // sensors on
  Wire.begin();
  uint8_t flags = (bootCount == 1) ? FLAG_BOOT : 0;
  if (!ads.begin()) flags |= FLAG_SELFTEST;
  ads.setGain(GAIN_ONE);                        // +-4.096 V
  tempSensor.begin();

  // ---- flow: count pulses during warm-up window (sensors settle meanwhile)
  pinMode(PIN_FLOW, INPUT_PULLUP);
  pulses = 0;
  attachInterrupt(digitalPinToInterrupt(PIN_FLOW), onPulse, FALLING);
  uint32_t t0 = millis();
  delay(WARMUP_MS);

  // ---- water temperature
  tempSensor.requestTemperatures();
  float tempC = tempSensor.getTempCByIndex(0);
  if (tempC < -50) { tempC = 25.0f; flags |= FLAG_SELFTEST; }

  // ---- analog probes
  float vPh = readVolts(0), vOrp = readVolts(1), vTurb = readVolts(2), vTds = readVolts(3);
  float ph = 7.0f + (vPh - PH_V7) / PH_SLOPE;
  float orp = (2.0f - vOrp) * 1000.0f + ORP_OFFSET_MV;
  float ntu = max(0.0f, TURB_A * vTurb * vTurb + TURB_B * vTurb + TURB_C);
  float comp = 1.0f + 0.02f * (tempC - 25.0f);
  float vc = vTds / comp;
  float tds = TDS_K * (133.42f * vc * vc * vc - 255.86f * vc * vc + 857.39f * vc) * 0.5f;

  while (millis() - t0 < FLOW_WINDOW_MS) delay(10);
  detachInterrupt(digitalPinToInterrupt(PIN_FLOW));
  float lpm = (pulses / FLOW_K_PULSE_PER_L) * (60000.0f / FLOW_WINDOW_MS);

  digitalWrite(PIN_SENSOR_PWR, LOW);            // sensors off before radio burst

  // ---- battery
  float vbat = analogReadMilliVolts(PIN_VBAT) * 2.0f;
  if (vbat < 3450) flags |= FLAG_LOW_BATT;

  // ---- 17-byte payload, big-endian (matches jalnetra/codec.py FMT ">BHBHhHHHhB")
  uint8_t p[17];
  p[0] = NODE_ID;
  put16(p, 1, seq);
  p[3] = flags;
  put16(p, 4, (uint16_t)constrain(lroundf(ph * 100), 0, 1400));
  put16(p, 6, (uint16_t)(int16_t)constrain(lroundf(orp), -2000, 2000));
  put16(p, 8, (uint16_t)constrain(lroundf(ntu * 10), 0, 65535));
  put16(p, 10, (uint16_t)constrain(lroundf(tds), 0, 65535));
  put16(p, 12, (uint16_t)constrain(lroundf(lpm * 10), 0, 65535));
  put16(p, 14, (uint16_t)(int16_t)lroundf(tempC * 10));
  p[16] = (uint8_t)constrain(lroundf((vbat - 2500) / 10.0f), 0, 255);

  // ---- transmit
  LoRa.setPins(PIN_LORA_NSS, PIN_LORA_RST, PIN_LORA_DIO0);
  if (LoRa.begin(LORA_FREQ)) {
    LoRa.setSpreadingFactor(LORA_SF);
    LoRa.setSignalBandwidth(125E3);
    LoRa.setCodingRate4(5);
    LoRa.setTxPower(LORA_TX_DBM);
    LoRa.enableCrc();
    LoRa.beginPacket();
    LoRa.write(p, sizeof(p));
    LoRa.endPacket();
    LoRa.sleep();
  }
  seq++;

  // ---- deep sleep (~0.12 mA board-level incl. regulator)
  esp_sleep_enable_timer_wakeup((uint64_t)SLEEP_MINUTES * 60ULL * 1000000ULL);
  esp_deep_sleep_start();
}

void loop() {}
