<table>
<tbody>
<tr>
  <td><p>Disorders of body schema, a mental representation of the body that serves action, are central to various psychopathologies (e.g., eating disorders, schizophrenia) and medical conditions (e.g., obesity, surgeries, neurology). For these patients, there is a deviation of body representation from the boundaries of their biological body.</p></td>
  <td><img src="https://github.com/user-attachments/assets/c1d71623-6064-45bf-bd97-185d389aea4f"  width="1500"></td>

</tr>
</tbody>
</table>



However, those deviations are often overlooked and therefore inadequately addressed, to the detriment of patients' psychotherapies. The difficulty lies in conducting quantitative assessments of body schema, which require not only questionnaires but also quantitative motor skill tests. In medical and psychological research, body schema has been assessed using passability tasks, that consist in examining the shoulder rotation of individuals when walking through adjustable apertures.



#### The RECORD setup
For the aperture task test, the subject is wearing a _Shoulder Pod_ and is asked to walk toward a doorway. Each wireless TBS is positioned at fixed distance to the door. The full setup, presented consists of :
 
* 1 wireless IMU _Shoulder Pod_
* 4 wireless custom made _Through-Beam Sensor Pod_ (_TBS Pod_)
* 1 Wifi Router 
* A slidding door

An application running on a Docker container, synchronizes the pods, acquires, records the data and computes the shoulder rotation angles.

![Image](https://github.com/user-attachments/assets/c73c30dd-fafa-49e5-9cf6-a5f5989c2e81)



### Shoulder Pod
![Image](https://github.com/user-attachments/assets/032a15ab-3642-44b4-9317-4dfbb52f3cac)

#### List of Components
<table>
<tbody>
<tr>
  <th>Parts</th>
  <th>Number</th>
  <th>Provider</th>
</tr>
<tr>
  <td>Seeed Studio XIAO ESP32C3</td>
  <td>1</td>
  <td><a href="https://eu.mouser.com/ProductDetail/Seeed-Studio/102010636?qs=sGAEpiMZZMuqBwn8WqcFUjg%252BG1hlSMIP0F1ZoZTWMTaX2wYW3M%252Bc5g%3D%3D"><img src="https://upload.wikimedia.org/wikipedia/commons/9/9a/Mouser_Electronics_logo.svg" alt="Mouser" height="40"></a></td>
</tr>
<tr>
  <td>IMU BNO055 </td>
  <td>1</td>
  <td><a href="https://eu.mouser.com/ProductDetail/Adafruit/2472?qs=N%2F3wi2MvZWDmk8dteqdybw%3D%3D"><img src="https://upload.wikimedia.org/wikipedia/commons/9/9a/Mouser_Electronics_logo.svg" alt="Mouser" height="40"></a></td>
</tr>
  <td> LiPo Battery 3.7 V 400 mAh</td>
  <td>1</td>
  <td><a href="https://eu.mouser.com/ProductDetail/Adafruit/3898?qs=byeeYqUIh0NUfilp6w7tLA%3D%3D"><img src="https://upload.wikimedia.org/wikipedia/commons/9/9a/Mouser_Electronics_logo.svg" alt="Mouser" height="40"></a></td>
</tr>
<tr>
  <td> 3M Red dot elecrode 2560 </td>
  <td>1</td>
  <td><a href="https://www.distrimed.com/product_info.php?products_id=6057"><img src="https://hygie31.com/wp-content/uploads/2024/07/logo-distrimed-1024x290.png" alt="distrimed" height="40"></a></td>
</tr>
</tr>
  <td> Switch SK12D07VG3 </td>
  <td>1</td>
  <td><a href="https://lcsc.com/product-detail/Slide-Switches_SHOU-HAN_C431547.html"><img src="https://static.lcsc.com/feassets/pc/images/headIcons/logo-s.png" alt="LCSC" height="40"></a></td>
</tr>
<tr>
  <td> NeoPixel 5050 RGB </td>
  <td>1</td>
  <td><a href="https://www.adafruit.com/product/1655"><img src="https://upload.wikimedia.org/wikipedia/commons/9/90/Adafruit_logo.svg" alt="Adafruit" height="40"></a></td>
</tr>
</tbody>
</table>

#### PCB
<img src="https://github.com/user-attachments/assets/33761b80-6aea-450c-b799-0cbfd4452944"  width="300">
<img src="https://github.com/user-attachments/assets/8a9a5efa-ac91-4c7f-82e7-f89a9f0467e9"  width="300">


### Through Beam Detector Pod List of Components
<table>
<tbody>
<tr>
  <td>Seeed Studio XIAO ESP32C3</td>
  <td>https://eu.mouser.com/ProductDetail/Seeed-Studio/102010636?qs=sGAEpiMZZMuqBwn8WqcFUjg%252BG1hlSMIP0F1ZoZTWMTaX2wYW3M%252Bc5g%3D%3D</td>
</tr>
<tr>
  <td>LiDAR POL4079 </td>
  <td> </td>
</tr>
  <td> LiPo Battery 3.7 V 400 mAh </td>
  <td> https://eu.mouser.com/ProductDetail/Adafruit/3898?qs=byeeYqUIh0NUfilp6w7tLA%3D%3D </td>
</tr>
</tbody>
</table>

