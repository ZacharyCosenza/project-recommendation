# Event recommender

Every week I want to know what's going on near me that I'd actually like. The listings are out there but scattered across venue sites, ticketing pages and SEO junk, and none of them know my taste. So this project does two things: (1) an agent builds a catalog of events from web search, and (2) a small model learns my taste from a handful of 👍/👎 labels and picks what to show me next.

```mermaid
flowchart LR
    S["queries + city + dates"] --> A["Claude agent<br/>refines web searches"]
    A --> D["event descriptions"]
    D --> C["context<br/>(day + event type)"]
    D --> E["embedding of<br/>description + context"]
    C --> E
    E --> GP["Gaussian process<br/>classifier"]
    GP --> R["rank by P(like)"]
    R --> U["I label 👍 / 👎<br/>and edit context"]
    U --> GP
    U --> C
```

Everything below uses the real data behind the [live app](https://event-recommender.fly.dev): 265 events from 14 searches across Chicago, New York, Monterrey, Reno and Philadelphia.

# Building features with an agent

A search API gives you pages, not events. Ask for "comedy shows in Chicago this weekend" and you mostly get "47 best things to do" listicles. So Claude gets one tool (`brave_search`) and one instruction: start broad, and if the results are vague, search again with sharper keywords (a date, a venue, a series name it just learned) until it finds specific events. Out comes a description per event:

```json
{
  "https://www.choosechicago.com/event/jerry-seinfeld-live-2/":
    "Jerry Seinfeld Live! at The Chicago Theatre (175 N. State St.). Fri Oct 9, 2026, 7:00 PM. His tour site and Ticketmaster also list two shows on Sat Oct 10, at 5:00 PM and 8:00 PM."
}
```

That description is the main feature the model sees, so prompting the agent is basically feature engineering. It costs ~1.5¢ of Claude tokens per event.

**Context.** On top of the description, each event gets a short *context*. A cheap second pass (Claude Haiku, batches of 20 descriptions, about 4¢ per thousand events) pulls out the event's first date and what kind of event it is. Python turns the date into a weekday, which is more reliable than asking the LLM for one, and anything it can't find stays blank. The Seinfeld show above becomes `Friday, comedy show`. I can edit any context in the app to add what the description can't know, like "date night" or "too far from home".

![Event types and weekdays extracted from the descriptions](docs/images/context.png)

A small local model (`all-MiniLM-L6-v2`) then turns `description + " Context: " + context` into a 384-number embedding. The model is never told which search found an event, but events from the same search still land near each other, which is what makes learning taste possible (like one event, its neighbors are good guesses).

![Average similarity between events, grouped by the search that found them](docs/images/similarity.png)

# Recommending with a Gaussian process classifier

I've only labeled a few dozen events, way too little for the recommenders behind big apps. Instead I use a Gaussian process classifier: it learns from a handful of labels, gives probabilities instead of a hard yes/no, and naturally becomes less confident about events unlike anything I've labeled. Every label I give makes the next pick better. The real model works in all 384 dimensions, so to make it visible I squeeze everything onto one axis: similarity to "Oktoberfest Chicago in Fulton Market", an event I liked.

**Step 1: labels.** Each dot is an event with its label: 👍 (+1), neutral (0, the default) or 👎 (−1). There's a pattern but it's noisy: in the least Oktoberfest-like third, dislikes beat likes 15 to 4, and in the most Oktoberfest-like third likes lead 23 to 12.

![Labeled events along the Oktoberfest-similarity axis](docs/images/step1_labels.png)

**Step 2: probabilities, not a score.** The labels are categories, so instead of regressing on −1/0/1 the model is a classifier: one Gaussian process per class (one-vs-rest), combined into P(dislike), P(neutral) and P(like) for every event. On this axis P(like) climbs from about 2% to 60% as events get more Oktoberfest-like. P(dislike) barely moves (~15%): my dislikes aren't about this direction at all, which is exactly why the real model needs all 384 dimensions.

![Class probabilities along the Oktoberfest axis](docs/images/step2_classifier.png)

**Step 3: does the context help?** To check, I cross-validated the classifier on embeddings of the description alone vs. description + context. Honestly, not yet. Both find likes about equally well (ROC-AUC 0.89 vs 0.88) and dislikes a hair better with context (0.73 vs 0.74), all within noise. That makes sense: the auto-extracted context is pulled *from* the description, so it mostly repeats what's already there. Where it should pay off is the edits, the things only I know about an event.

![Cross-validated ROC-AUC with and without context](docs/images/step3_context_cv.png)

**Step 4: close the loop.** The app ranks every event by P(like): show the most likely 👍, I rate it, retrain, repeat. To check this works I replayed it on all 265 events (30 liked), hiding each rating until the model "showed" me that event. Starting from 5 random labels, ranking by P(like) found 11 likes in 40 picks vs 4 for random order, more than 2.5× as many. Right now the top unrated picks are a Chicago Oktoberfest street fest (54%), Don Giovanni at the Lyric Opera (45%) and the second Seinfeld show (35%).

![Liked events found over 40 picks](docs/images/step4_replay.png)

# The app

The app shows the same thing in 2D: events laid out by similarity (UMAP) over a heat map of P(like). Clicking an event opens a card with its context (editable) and highlights which words push P(like) up (blue) or down (red). In the events table I label with a single click on the 👍 or 👎 box and can edit any event's context in place; editing a context re-embeds that event immediately, and the next retrain picks it up.

![Event map: UMAP layout over a P(like) heat map](docs/images/event_map.png)
