# Episode 06: When training stops helping, search

This morning portfolio version two was the new default. It won fifty seven percent of games against version one. You asked me to keep improving the model, so this episode is about what I tried, what failed, and the one thing that worked.

First, a third round of training. Every specialist trained against the best specialists of round two, like a league. It made no difference. Halfway through, the new portfolio tied version two, fifty percent to fifty. So I stopped it.

Then four smaller ideas. An endgame rule that looks at the score: buy the last Province only if it wins, and never if it loses. It changed nothing, forty nine point eight percent. A rule to wait for enough money before buying Provinces. The evolved buy lists already do that, so the rule almost never fired. A bigger pool, with version one and version two specialists together. Again fifty percent. A longer trial before each game, to pick the right specialist. That one helped a little, half a point, and it is now on.

The lesson is that the specialists have reached what their strategy can express. A fixed list of what to buy, in three phases, can only get so good. More training of the same kind won't change that.

So I built a different kind of player on top of them, called rollout search. At every buy, it takes the specialist's choice, its next few choices, and buying nothing. For each one, it plays the rest of the game out three hundred times in the fast engine, with the cards it can't see shuffled at random. Then it picks the option that wins most often. But it only overrules the specialist when the difference is clear.

To make that possible, the engine learned to copy a game in the middle of a turn and continue it. A test checks that a copied game finishes exactly like the original.

The first try was worse, about forty four percent, because with only thirty playouts the differences are mostly luck. With three hundred playouts and a small margin, the search wins sixty four percent of games against the same specialist without search. That is over two thousand games, and it's the biggest gain in the whole project.

There is one condition. The search needs to know what is in the opponent's deck. When it had to imagine the opponent as a copy of its own deck, the gain disappeared. On Steam, though, the opponent's bar at the top of the screen says Added, and the card, every time they gain one. So the autopilot now reads that banner and keeps count of the opponent's cards. In simulation, even if it misses a quarter of them, the search still wins about fifty eight percent.

The new model is called portfolio version two s, with s for search. It is published, but it isn't the default yet. It has not played a real Steam game, and that game takes over the mouse, so it waits until you say go.

So what's next is simple. Watch a few real games with the search model. Check that the opponent count matches the card counter on screen. If it holds up, it becomes the default.
