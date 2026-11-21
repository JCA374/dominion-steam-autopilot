# Episode 08: Training all of the GA, and a better way to pick

Last episode I said the genetic algorithm had hit its ceiling. That was true for the specialists I tested, like Witch and Mine. But I had only tested a handful. You asked me to try all of them, and to move on wherever they stopped improving. So every specialist in the portfolio got one thousand more generations, and each one was measured on games no run had seen.

The result: seventeen of the twenty six specialists got better. Moneylender gained the most. It now beats its old version about two games in three, and it went from sixty two to seventy one percent against the simple bots. Library, Harbinger, Laboratory and Council Room also gained clearly. Nine did not move at all, among them Witch, Chapel, Bandit and the generalist. The biggest gains came from specialists that started weak, like Moneylender and Library. The strongest ones, like Witch and Gardens, were already at the ceiling.

Then the surprise. When I put all seventeen improved specialists into the baseline, it gained only one and a half points against the reinforcement learning network. Why so little? The baseline plays only one specialist per kingdom, the strongest one that fits. Most of the specialists that improved are ones it rarely picks.

So the real question was the picking itself. The baseline the other models are measured against chooses a specialist by a fixed rule: the one that scores best against the simple bots. That turns out to be a poor guide. On the most common kind of kingdom it played Militia, while Merchant and Bandit beat Militia head to head about sixty two percent of the time.

I tried three better rules. A fixed table of head to head results helped a little. The best rule is simple. When the game starts, every specialist that fits the kingdom plays every other one on that exact kingdom, about a hundred quick games each. The one with the best average plays the real game. It takes about half a second.

On its own, that new rule adds about five points against the reinforcement learning network. Together with the seventeen better specialists, the new baseline scores about six points more than the old one, on a thousand fresh kingdoms. The old baseline won about thirty percent of its games against the network on the kingdoms where the pick changed. The new one wins close to thirty nine. Going from a hundred to three hundred trial games per pair added nothing, so that rule has plateaued too.

One honest caveat. The portfolio the autopilot plays on Steam already had a good picker. It plays trial games against the bots at the start of each game. Head to head, my new rule did not beat it. So the released player was already about this strong. The gain is for the weaker baseline used in the comparisons. I have not swapped that baseline yet, because every past result was measured against the old one. That is your call.

Meanwhile, the Play page on the reinforcement learning site changed a lot. It now works on a phone. You can switch on a win probability and a hint from the strongest network, which glows on the card it would pick. There is an undo button, a review after the game, and a Learn page with the model's habits and a quiz. A gameplay spec file holds all the settings, so you can change them later.

What's next: decide whether the baseline gets the new picker, and stop training the genetic algorithm. Every part of it has now either improved or flattened out.
