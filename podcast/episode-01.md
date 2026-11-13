# Episode 01: From a simulator to the Steam game

You asked a simple question. Could the Dominion strategy from your genetic algorithm project play the real Dominion game on Steam? This episode is the story of how it got there, and what was in the way.

The strategy comes from a project that evolved Dominion players inside its own small game engine. It played thousands of games, kept the winners, and mixed and mutated them. What came out is one small file of lists. What to buy early, in the middle, and late. Which action cards to play first. What to trash with a Chapel. In its own engine it beats the simple Big Money strategy about three games in four, and it wins just over half its games against its own hall of fame.

The Steam game is a different world. It is a closed Windows program that runs on Linux through Proton. There is no way for a bot to plug in. No interface, no mods, and no file that lists the cards in your hand. So the only way in was the way a person plays. Look at the screen, decide, and click.

The first worry was the desktop. Your machine runs Wayland, and Wayland is built to stop one program from watching another. But Proton runs the game as an old style X eleven window, inside a layer called X Wayland. Inside that layer, a program can still copy a window's pixels. The first screenshot came back in about a twentieth of a second, with no permission dialog. A test click on the play treasures button worked too. That was the moment the project became possible.

The second piece was knowing the kingdom. Here the game helped. It saves each local game as a small text file, and one line of it lists the ten kingdom cards. So the autopilot never has to recognise the supply. It reads the list, then finds each pile on screen by its name.

Reading the screen is done by a text recognition model called Rapid O C R. It runs on the processor, on a few small regions. The prompt line says what the game wants. The hand shows the cards. And three counters show actions, coins and buys.

Then came the first surprise. The click on the Silver pile did nothing. It turned out the game only accepts a click after the pointer has hovered over the target for a moment. So every click now glides in, rests for about a third of a second, and then presses. After that, buying worked.

The second surprise was timing. At the start of a turn, the game animates, and clicks made during that animation are lost. Smithy sometimes needed three clicks. The fix was to wait for a still screen, and to wait until the actions counter shows at least one.

For the decisions, the autopilot doesn't copy the strategy's rules. It builds a stand-in game state from what it read, and calls the original buy function from the training engine. So the purchases on Steam are exactly the purchases the evolved strategy makes in training.

The first full game was against the very easy computer opponent. The autopilot bought Silver, Smithy, three Golds, and then Province after Province. It won fifty eight to eight, in twenty one turns, without any help.

That is where the first episode ends. The next one goes through every part in detail. The one after that is about what the recorded games revealed, including two bugs that only showed up in real play.
