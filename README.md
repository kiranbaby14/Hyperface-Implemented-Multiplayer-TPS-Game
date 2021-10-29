# Hyperface-Implemented-Multiplayer-TPS-Game
A multiplayer TPS game made using unity engine and input controls given via facial gestures.

<br/>

## Screenshot

![p2](https://user-images.githubusercontent.com/50899339/137939021-6e37c875-fdaf-4e2a-a832-90bccedd0e40.jpg)

<br/>

## Implementations

* ### Gender Detection 
  Gender setection is implemented by training a model on CNN algorithm. This mechanism is used in the game to select the character according to the gender of the player.
* ### Face Detection 
  Face detection is achieved using DLib library and OpenCV.
* ### Landmark Localization 
  Landmark Localization  is done using DLib library and OpenCV. This feature is also used in the implementation of mouth opening detection and eye blinking detection which are used inside the game to make the player perform certain actions.
* ### Multiplayer Game
  The Game is made using unity engine and multiplayer mechanism is implemented on top of photon server.

