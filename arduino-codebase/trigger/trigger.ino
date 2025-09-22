#include <WiFi.h>
#include <string>
#include <vector>


// La while loop d'écoute du client est estimé à 33 ms  de durée

// // Déclaration de variables globales
unsigned long startMillis;
unsigned long currentMillis;
const unsigned long period = 35;
const int COUNT_TIME_DISCONNT = 3000 / period;  // number of client while loop to wait until we get 3s of elapstime

// // Identifiants Wi-Fi
const char *hostname = "TRIGGER_SENSOR_04";
const char *ssid = "TP-Link_72B8";
const char *password = "Record2025!";
int socket = 64388;

// // Pins des LEDs et capteur
const uint8_t led_R = D0;
const uint8_t led_G = D1;
const uint8_t led_B = D2;
const uint8_t trig_pin = D3;
const uint8_t sensor_pin = D10;

class RGBLed {
private:
  const uint8_t redPin;
  const uint8_t greenPin;
  const uint8_t bluePin;

public:
  RGBLed(uint8_t redPin, uint8_t greenPin, uint8_t bluePin)
    : redPin(redPin), greenPin(greenPin), bluePin(bluePin) {
  }

  void begin() {
    pinMode(redPin, OUTPUT);
    pinMode(greenPin, OUTPUT);
    pinMode(bluePin, OUTPUT);
    turnOff();
  }

  void setColor(uint8_t redValue, uint8_t greenValue, uint8_t blueValue) {
    digitalWrite(redPin, redValue);
    digitalWrite(greenPin, greenValue);
    digitalWrite(bluePin, blueValue);
  }

  // Méthodes pour des couleurs prédéfinies
  void setRed() {
    setColor(HIGH, LOW, LOW);
  }

  void setGreen() {
    setColor(LOW, HIGH, LOW);
  }

  void setBlue() {
    setColor(LOW, LOW, HIGH);
  }

  void setYellow() {
    setColor(HIGH, HIGH, LOW);
  }

  void setCyan() {
    setColor(LOW, HIGH, HIGH);
  }

  void setMagenta() {
    setColor(HIGH, LOW, HIGH);
  }

  void setWhite() {
    setColor(HIGH, HIGH, HIGH);
  }

  void switch_color() {
    setRed();
    delay(10);
    setGreen();
    delay(10);
  }

  void switch_color_wifi_fail() {
    setRed();
    delay(500);
    setBlue();
    delay(500);
  }
  // Méthode pour éteindre la LED
  void turnOff() {
    setColor(LOW, LOW, LOW);
  }
};



class Sensor {
private:
  const uint8_t sensorPin;
  const uint8_t trigPin;
  float limitMaxRange;
  const float offsetMin;

public:
  float m_distance = 0.0;
  float threshMax;
  float threshMin;
  bool triggered;

  Sensor(uint8_t sensorPin, uint8_t trigPin, float offsetMin = 1000.0, float limitMaxRange = 3000.0)
    : sensorPin(sensorPin), trigPin(trigPin), threshMax(2000.0), threshMin(20.0),
      offsetMin(offsetMin), limitMaxRange(limitMaxRange), triggered(false) {
    pinMode(sensorPin, INPUT);
    pinMode(trigPin, INPUT);
  }

  float measureDistance() {
    float t = pulseIn(sensorPin, HIGH);
    float distance = (t - offsetMin) * 4;
    return (t == 0 || t > limitMaxRange) ? limitMaxRange : distance;
  }

  bool checkProximity(float distance) {
    return distance < threshMax && distance >= threshMin;
  }

  void set_threshMax(float value) {
    Serial.print("Update thresMax value from: ");
    Serial.print(threshMax);
    Serial.print(", to: ");
    threshMax = value;
    Serial.println(threshMax);
  }

  void set_threshMin(float value) {
    Serial.print("Update thresMin value from: ");
    Serial.print(threshMin);
    Serial.print(", to: ");
    threshMin = value;
    Serial.println(threshMin);
  }

  void auto_scale_treshMax() {
    float distance = measureDistance();
    set_threshMax(distance - 10.0);
  }

  String get_info() {
    float distance = measureDistance();
    String info_msg = "";
    info_msg += "Distance=";
    info_msg += distance;
    info_msg += " ";
    info_msg += "Proximity=";
    info_msg += checkProximity(distance);
    info_msg += " ";
    info_msg += "TreshMax=";
    info_msg += threshMax;
    info_msg += " ";
    info_msg += "TreshMin=";
    info_msg += threshMin;
    info_msg += " ";
    info_msg += "limitMaxRange=";
    info_msg += limitMaxRange;
    info_msg += "\n";
    return info_msg;
  }
};

class MyServer {
private:
  const char *ssid;
  const char *password;
  const char *hostname;
  int socket;
  WiFiServer server;
  Sensor sensor;
  RGBLed *m_rgb_led_ptr = nullptr;
  int m_CLIENT_DISCONNT_CPT = 0;
  int m_FALSE_POSITIVE_CPT = 0;
  bool m_wifi_disconnected_flag = false;

public:
  MyServer(const char *ssid, const char *password, const char *hostname, int socket,
           uint8_t sensorPin, uint8_t trigPin,
           RGBLed *rgb_led)
    : ssid(ssid), password(password), hostname(hostname), socket(socket), server(socket),
      sensor(sensorPin, trigPin), m_rgb_led_ptr(rgb_led) {}

  void begin() {
    setUpWifi();
  };

  void setUpWifi() {
    WiFi.mode(WIFI_STA);
    WiFi.setHostname(hostname);
    WiFi.begin(ssid, password);
    Serial.print("Hostname: ");
    Serial.println(hostname);
    Serial.print("Connecting to WiFi...");
    while (WiFi.status() != WL_CONNECTED) {
      m_rgb_led_ptr->switch_color_wifi_fail();
      Serial.print(".");
    }
    Serial.println("\nConnected!");
    Serial.print("IP Address: ");
    Serial.print(WiFi.localIP());
    Serial.print(":");
    Serial.print(socket);
    Serial.println();
    server.begin();
    m_rgb_led_ptr->setBlue();
  }

  void ReconnectToWiFi() {
    server.end();
    server.begin();  // restart the server
    m_rgb_led_ptr->setBlue();
  }

  void checkWifiConnectionLoop() {
    while (WiFi.status() != WL_CONNECTED) {
      m_rgb_led_ptr->switch_color_wifi_fail();
      Serial.print(".");
      m_wifi_disconnected_flag = true;
    }
    if (m_wifi_disconnected_flag == true)
    {
      ReconnectToWiFi();
      m_wifi_disconnected_flag = false;
    }
  }


  void handleClient() {
    WiFiClient client = server.available();
    if (client) {
      Serial.println("Client connected.");
      m_rgb_led_ptr->setGreen();
      String message = "";
      while (client.connected()) {
        currentMillis = millis();
        unsigned long elapseMillis = currentMillis - startMillis;
        if (elapseMillis >= period) {
          if (client.available()) {
            char c = client.read();
            if (c == '\n') {
              message.trim();
              processClientMessage(client, message);
              message = "";
            } else {
              message += c;
            }
          }
          checkSensorState(client);
          checkSensorRearm();
          if (check_push_client_disconnt()) {
            break;
          }
          startMillis = currentMillis;
        }
        checkWifiConnectionLoop(); // Check WiFi connection status
      }
      client.stop();
      Serial.println("Client disconnected.");
      String msg = "Sensor rearmed du to client disconnection";
      sensorRearm(msg);
      m_rgb_led_ptr->setBlue();
    }
  }

  void error_cmd(WiFiClient &client, String &message) {
    Serial.printf("Reveived command : %s\n", message.c_str());
    client.println("ERROR: UNKNOWN COMMAND");
  }

  void processClientMessage(WiFiClient &client, String &message) {
    int separator_index = message.indexOf('#');
    if (separator_index == -1) {
      if (message == "REQ_STATUS") {
        client.println(sensor.triggered ? "DONE" : "READY");
      } else if (message == "REQ_RESET") {
        checkSensorRearm(true);
        client.println("ACK_RESET");
      } else if (message == "ACK_TRIGGERED") {
        sensor.triggered = true;
        Serial.println("Client acknowledge beeing triggered !");
      } else if (message == "REQ_INFO") {
        String reply = "ACK_INFO " + sensor.get_info();
        client.println(reply);
      } else if (message == "REQ_AUTO_TRESH_MAX") {
        sensor.auto_scale_treshMax();
        client.println("ACK_THRESH_MAX");
      } else {
        error_cmd(client, message);
      }
    } else {
      // Handling pair "key#value" messages
      String key = message.substring(0, separator_index);
      float value = message.substring(separator_index + 1).toFloat();
      Serial.printf("Reveived command : key=%s, value=%s\n", key.c_str(), String(value).c_str());

      if (key == "TRESH_MAX") {
        Serial.print("Given value= ");
        Serial.println(value);
        sensor.set_threshMax(value);
        client.println("ACK_THRESH_MAX");
      } else if (key == "TRESH_MIN") {
        Serial.print("Given value= ");
        Serial.println(value);
        sensor.set_threshMin(value);
        client.println("ACK_THRESH_MIN");
      } else {
        error_cmd(client, message);
      }
    }
  }

  void checkSensorState(WiFiClient &client) {
    float distance = sensor.measureDistance();
    if (sensor.checkProximity(distance) && !sensor.triggered) {
      m_FALSE_POSITIVE_CPT++;
      if (m_FALSE_POSITIVE_CPT == 2) {
        client.printf("TRIGGERED=%s\n", String(distance).c_str());
        m_rgb_led_ptr->setRed();
      }
    } else if (m_FALSE_POSITIVE_CPT > 0) {
      m_FALSE_POSITIVE_CPT--;
    } else {
      client.printf("IDLE\n");
    }
  }

  void sensorRearm(String &msg) {
    sensor.triggered = false;
    m_rgb_led_ptr->setGreen();
    Serial.println(msg);
  }

  void checkSensorRearm(bool req_client = false) {
    if (digitalRead(trig_pin) == HIGH && sensor.triggered) {
      String msg = "Rearmed manually";
      sensorRearm(msg);
    }
    if (req_client == true) {
      String msg = "Rearmed from client request";
      sensorRearm(msg);
    }
  }

  bool check_push_client_disconnt() {
    if (digitalRead(trig_pin) == HIGH) {
      m_CLIENT_DISCONNT_CPT++;
      m_rgb_led_ptr->switch_color();
    } else if (m_CLIENT_DISCONNT_CPT > 0) {
      m_CLIENT_DISCONNT_CPT--;
    }
    return m_CLIENT_DISCONNT_CPT >= COUNT_TIME_DISCONNT;
  }

  void loop() {
    handleClient();
    checkWifiConnectionLoop(); // Check WiFi connection status
  }
};


RGBLed rgb_led(led_R, led_G, led_B);
MyServer my_server(ssid, password, hostname, socket,
                   sensor_pin, trig_pin,
                   &rgb_led);


void setup() {
  // put your setup code here, to run once:
  rgb_led.begin();
  rgb_led.setWhite();
  Serial.begin(115200);
  startMillis = millis();
  my_server.begin();
}

void loop() {

  my_server.loop();
}
