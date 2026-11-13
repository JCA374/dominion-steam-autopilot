# How the Dominion autopilot works

## intro

This film explains, step by step, how a strategy that was evolved in a simulator ends up playing the real Dominion game on Steam. Nothing here is a mock-up. Every frame of the game comes from a recording of the autopilot playing, and every box, click and log line on screen is what the program itself saw and did at that moment.

## problem

Start with the problem. The strategy was evolved by a genetic algorithm. It played many thousands of games inside its own small Dominion engine, written in Python and C. The Steam game is a different program. It is a closed Windows game that runs on Linux through Proton. It has no interface for bots, no mods, and no file that shows the cards in your hand. So the only way in is the same way a person plays it. Look at the screen, decide, and click.

## pipeline

The autopilot therefore has three parts. The eyes take a screenshot of the game window and read the text on it. The brain turns what the eyes read into a decision, using the evolved strategy. The hands move the mouse and click. A loop runs these three parts over and over. Each pass handles exactly one thing the game asks for, such as play an action, play your treasures, or buy a card.

## screenshots

The eyes start with a screenshot. On this desktop the screen is run by Wayland, which normally blocks one program from looking at another. But Proton runs the game as an old style X eleven window, inside a compatibility layer called X Wayland. Inside that layer, a program can still ask for the pixels of a window. The autopilot finds the window named Dominion and copies its pixels into memory. One screenshot takes about a twentieth of a second. No permission dialog appears.

## ocr

Next it reads the screenshot. The autopilot uses an optical character recognition model called Rapid O C R, which runs locally on the processor. It does not read the whole screen. It reads a few fixed regions, drawn here as blue boxes. One box holds the prompt line, the sentence under the supply that says what the game wants. One holds the hand, where each card has its name on a banner. One holds the three counters for actions, coins and buys. And one holds the buttons on the right.

## matching

Optical character recognition makes small mistakes. It might read Estate as State, or Council Room as one word. So every name it reads is matched against the list of cards in this game, and the closest name wins if it is at least eighty percent similar. Numbers are read as plain digits. The province count sits on a pile that dims when you cannot afford it, so when that number can't be read, the autopilot keeps the last count it saw. One mistake was nastier. The card at the far left of the hand is tilted, and once in a while the reader decided it was upside down. Smithy came out as a h l i w s. In the recordings, that made the autopilot skip three Smithy plays. The reader has a check that turns text the right way up, and on that tilted card it guessed wrong. Hand cards are never upside down, so that check is now off for the hand. And any word that still matches no card is read once more, on its own.

## kingdom

The ten kingdom cards don't need to be read from the screen at all. Dominion saves every local game as a small file inside the Proton folder. That file is plain X M L, and one line of it lists the kingdom, for example chapel, moat, merchant, smithy and witch. It also names the opponent. So when a game starts, the autopilot reads that file and knows the exact supply. The cards are still found on screen by their names, because their positions can change from game to game.

## strategy

Now the brain. The evolved strategy is a small file of lists. It splits the game into an early, a middle and a late phase. In this model the early phase is turns one and two. The middle runs until only two provinces are left, or until turn sixteen, whichever comes first. Each phase has its own buy list, in order. Early, it wants a Smithy, then a Duchy, then Silver, then Merchant, and then it passes. In the middle it wants Province, then Gold, then Silver. Late, Province first, then Duchy, then Moneylender, then Estate.

## buying

To buy, the autopilot does something careful. It doesn't copy those rules into new code. It builds a stand-in game state with the coins and buys it just read, and the cards it owns. Then it calls the original buy function from the strategy's own engine. Whatever that function buys is the plan. That way the Steam autopilot makes exactly the purchases the evolved strategy would make in training. That includes two hidden rules. One caps how many copies of each card to own. The other refuses a terminal action card when the deck has no spare actions to play it.

## unknown

There is a catch. The strategy only ever learned fifteen kingdom cards from the base set. Steam deals kingdoms from hundreds of cards. Any card the strategy never saw is simply left out of the stand-in supply. So the autopilot never buys it, and never needs to know how to play it. In the recorded games, cards like Harbinger, Workshop and Remodel were ignored this way.

## actions

Playing actions follows the same idea. The strategy keeps two ordered lists. Non terminal cards, which give back an action, are played first, Merchant before Market before Village. Then terminal cards, such as Smithy. A Moneylender is skipped when there is no Copper to trash. After each card it plays, the autopilot takes a fresh screenshot, because the hand has changed.

## clicks

Then the hands. The autopilot sends mouse events through an X eleven extension called X test, which X Wayland passes on to the game. The first version clicked instantly, and the game ignored the clicks on cards. It turned out the game only accepts a click after the pointer has hovered over the target for a moment. So every click now glides the last few pixels, rests for about a third of a second, then presses and holds briefly. The gold ring on screen marks each click.

## timing

Timing caused the other early bug. At the start of a turn, the game animates. The hand slides in, and the opponent's last card still shows in the middle. A click during that animation is lost. So before reading anything, the autopilot takes two screenshots a third of a second apart, and waits until they are almost identical. And in the action phase, it also waits until the actions counter shows at least one. With those two rules, Smithy plays on the first click.

## empty

The second game it played found one more bug. The Duchy and the Estate piles ran out. The strategy still asked for them, so the autopilot clicked an empty pile, nothing happened, and the turn ended without a purchase. Provinces stayed at two from turn twenty five to turn fifty two. In training this never came up, because the simulator knows when a pile is empty. Now, after every buy click, the autopilot checks whether the coins and buys changed. If they didn't, that pile is empty. It notes this and asks the strategy again without that card.

## turn

Here is one whole turn at real speed. The prompt says play action cards. The eyes read the hand, the brain picks Smithy, and the hands click it. Three new cards arrive. The prompt changes to play treasure cards, and it presses the button. Then buy a card. It reads the coins, asks the strategy, and clicks the pile. With one buy, the turn ends by itself, and the opponent takes its turn.

## memory

The autopilot also keeps a little memory. It counts its turns, so it knows which phase it is in. It tracks what it has bought and trashed, which the buy rules need. It saves this after every turn to a small file named after the game's seed. If you stop it and start it again in the middle of a game, it picks up where it was.

## ending

When the last province goes, the game shows a score screen. The prompt line is empty then, so the autopilot looks for the Rematch button. When it finds it, it reads both scores and logs the result. With the games option, it then presses Rematch. It waits for the new save file to appear, reads the new kingdom, and plays again.

## limits

Finally, the limits. It only knows fifteen cards. If the game asks something it doesn't understand, for example a choice from a card it doesn't know, it saves a screenshot and waits for you to answer by hand. It assumes the window keeps roughly its size. And it should only play against the built in computer opponents, never against people online. Inside those limits, it plays a complete game on its own, from the first treasure to the score screen.
