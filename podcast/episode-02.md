# Episode 02: How it sees, decides and clicks

This episode goes through the autopilot one part at a time, in the order it works during a turn. If you want to see it, the film called How it works shows the same steps on recorded footage.

Everything runs in one loop. Each pass handles exactly one thing the game asks for. And each pass has three parts. The eyes, the brain, and the hands.

The eyes start with a screenshot of the game window. Before it reads anything, the autopilot takes a second screenshot a third of a second later. It compares the two. If they differ, the game is still moving, so it waits and tries again. Only a still screen gets read.

Then it reads the prompt line. That is the sentence just under the supply. It says things like play action cards, play treasure cards, or buy a card. The prompt decides which handler runs next. A prompt it doesn't recognise is saved as a picture, and the autopilot waits for you.

If the prompt asks for actions, it reads the hand. Every card has its name on a banner at the top, and the text reader returns each name with its position. Those positions are where it will click later. The names are matched against the cards in this game. A small reading mistake, like State instead of Estate, still matches, as long as the two are eighty percent alike or more.

Next it reads the counters. They are three small numbers left of the hand, for actions, coins and buys. In the action phase it waits until the actions counter shows at least one, because a zero means the turn is still starting.

Now the brain. The evolved strategy splits the game into three phases. In this model the early phase is just the first two turns. The middle lasts until only two provinces are left, or until turn sixteen. After that comes the late phase. Each phase has its own buy list. Early, it wants Smithy above everything. In the middle, Province, then Gold, then Silver. Late, Province, then Duchy.

For actions it keeps two lists. Cards that give back an action, like Merchant or Market, are played first. Then the terminal cards, like Smithy. It skips a Moneylender when there is no Copper in hand to trash.

Buying is the part I am most careful about. The autopilot doesn't re-write the buy rules. It builds a stand-in game state. That state holds the coins and buys it just read, the cards it owns, and a supply without the cards the strategy never learned. Then it calls the buy function from the original training engine, and uses whatever that function buys. This brings along two rules that are easy to forget. One caps how many copies of a card to own. The other refuses a terminal action card when the deck has no spare actions to play it.

The hands come last. Clicks are sent through an X eleven extension called X test. The game only accepts a click after a short hover. So the pointer glides in, rests for about a third of a second, and then presses and holds. Hand cards are clicked a little below their name, on the card itself, because a hovered card lifts up.

Between turns, the autopilot keeps a small memory file named after the game's seed. It holds the turn number, the provinces left, the cards it owns, and any piles it found empty. If you stop it in the middle of a game and start it again, it reads that file and carries on.

At the end, the score screen appears. The autopilot reads the points, saves a picture of the screen, and stops. Or, if you asked for more games, it presses rematch and starts again with the new kingdom.

That is the whole machine. In the next episode, I'll tell you what the recorded games revealed, and how you can run it yourself.
