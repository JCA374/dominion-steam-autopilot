# Episode 03: What three games taught, and how to run it

In the first episode, the autopilot won its first game fifty eight to eight. That game was clean. The next two games were recorded, frame by frame, to make the films. And the recordings found problems that a clean game had hidden.

The first problem was a card that was read upside down. The card at the far left of your hand is drawn tilted. Once in a while, the text reader decided it was upside down, and turned it over before reading. Smithy came out as a h l i w s. With no action card found, the autopilot ended its action phase and bought nothing with two coins. That happened twice in the second game.

My first fix was wrong, and I want to be honest about that. I read the unmatched word again, turned upside down. It passed my test, but my test had flipped the picture by hand. On the real frame, the fix made it worse. In the third game, Smithy was skipped again on turn six. The real fix was simpler. Hand cards are never upside down, so the reader's turning check is now off for the hand. After that, every Smithy was read and played.

The second problem was stranger. In the second game, the provinces stayed at two for twenty seven turns. The autopilot kept buying a Duchy or an Estate every turn. But those piles were empty. Clicking an empty pile does nothing, so every turn ended without a purchase. The strategy had never seen this in training, because its simulator always knows when a pile runs out. The fix is a small check. After every buy click, the autopilot compares the coins and buys with what they were before. If nothing changed, the pile is empty, and it asks the strategy again without that card. Restarted with that fix, it found both empty piles in two turns, went back to buying treasure, and won seventy six to seven.

In the third game the same check worked from the start. The Estate pile ran out on turn twenty five, and the autopilot noticed after one click. That game ended on three empty piles, and it won fifty five to thirty five.

The long games also showed a weak spot in the strategy itself. In its late phase it prefers Silver to Gold, and with one or two coins it still buys Copper. So a long late game fills the deck with weak cards, and eight coins become rare. In training this hardly mattered, because every simulated game stops at turn forty. That is a lesson for the next round of evolution, not a bug in the autopilot.

So, three games against the very easy computer player, and three wins.

If you want to run it yourself, it takes about five minutes. Clone the repository with its submodule. Make a Python virtual environment, and install the five packages from the requirements file. Start Dominion from Steam, and begin a game against the computer. Leave the window at its normal size and in plain view. Then run the pilot script, and keep your hand off the mouse. The film called How to run it shows every step, and the site lists every option.

Two habits help. Start with the dry option once. It reads and decides without clicking, so you can check the log against what you see. And if something goes wrong, look in the unknown folder. Anything the autopilot didn't understand is saved there as a picture.

That's where things stand. The autopilot plays whole games on its own. Its main limit now is the strategy, which only knows fifteen cards and has a weak late game. Both of those are jobs for the genetic algorithm, not for the eyes and hands.
