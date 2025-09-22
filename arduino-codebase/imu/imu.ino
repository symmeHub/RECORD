#include <WiFi.h>
#include <Adafruit_BNO055.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_NeoPixel.h>
// #ifdef __AVR__
//  #include <avr/power.h> // Required for 16 MHz Adafruit Trinket
// #endif

#define LEDPIN 16
#define BRIGHT_LVL 5

unsigned long startMillis;  //some global variables available anywhere in the program
unsigned long currentMillis;
unsigned long clientMillis;
unsigned long startclientMillis;
const unsigned long period = 6;  //the value is a number of milliseconds


// Définir vos identifiants Wi-Fi
const char *hostname = "IMU-BNO055-01";
const char *ssid = "TP-Link_72B8";
const char *password = "Record2025!";
const int nb_sub_sample = 1;
const int nb_of_data = 8;  // qw, qx, qy, qz, elapse_time, acc_x, acc_y, acc_z
const int size_of_data_array = nb_sub_sample * nb_of_data;
float data[size_of_data_array];
float timestamp;

// Led NeoPixel
Adafruit_NeoPixel NEOLED(1, LEDPIN, NEO_GRBW + NEO_KHZ800);

// Configuration du serveur TCP
WiFiServer server(64344);

// Initialisation du capteur BNO055
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);

imu::Vector<3> a;
imu::Quaternion quat;

void enable_external_antenna() {
  pinMode(3, OUTPUT);  // RF switch power on
  digitalWrite(3, LOW);
  pinMode(14, OUTPUT);  // select external antenna
  digitalWrite(14, HIGH);
}


void setup() {
  enable_external_antenna();
  Serial.begin(115200);
  NEOLED.begin();
  NEOLED.setPixelColor(0, NEOLED.Color(0, 0, 0, BRIGHT_LVL));
  NEOLED.show();
  startMillis = millis();
  // Initialisation du capteur
  if (!bno.begin()) {
    Serial.println("Impossible de détecter le BNO055");
    while (1)
      ;
  }
  bno.setExtCrystalUse(true);

  WiFi.mode(WIFI_STA);
  WiFi.config(INADDR_NONE, INADDR_NONE, INADDR_NONE, INADDR_NONE);
  WiFi.setHostname(hostname);

  // Connexion au Wi-Fi
  Serial.println("Connexion au Wi-Fi...");
  WiFi.begin(ssid, password);
  WiFi.setSleep(false);


  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }
  Serial.println("\nWi-Fi connecté");
  Serial.print("Adresse IP: ");
  Serial.println(WiFi.localIP());
  Serial.print("Hostname: ");
  Serial.println(WiFi.getHostname());
  Serial.print("RRSI: ");
  Serial.println(WiFi.RSSI());
  // Démarrage du serveur TCP
  server.begin();
  Serial.println("Serveur TCP démarré");
  NEOLED.setPixelColor(0, NEOLED.Color(0, 0, BRIGHT_LVL, 0));
  NEOLED.show();
}

void loop() {
  // Vérification des connexions clients
  WiFiClient client = server.available();
  if (client) {
    NEOLED.setPixelColor(0, NEOLED.Color(0, BRIGHT_LVL, 0, 0));
    NEOLED.show();
    Serial.println("Client connected");
    startclientMillis = millis();
    // Lecture des quaternions et envoi
    while (client.connected()) {
      currentMillis = millis();
      timestamp = currentMillis - startclientMillis;
      if (currentMillis - startMillis >= period) {
        for (int i = 0; i < nb_sub_sample; i++) {
          int index = nb_of_data * i;
          quat = bno.getQuat();
          a = bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);
          data[index] = quat.w();
          data[index + 1] = quat.x();
          data[index + 2] = quat.y();
          data[index + 3] = quat.z();
          data[index + 4] = timestamp;
          data[index + 5] = a.x();
          data[index + 6] = a.y();
          data[index + 7] = a.z();
        }
        client.write((uint8_t *)data, sizeof(data));
        startMillis = currentMillis;
      }
    }
    client.stop();
    Serial.println("Client disconnected");
    Serial.print("Timestamp=");
    Serial.println(timestamp);
    NEOLED.setPixelColor(0, NEOLED.Color(0, 0, BRIGHT_LVL, 0));
    NEOLED.show();
  }
}
