# Theft Detection Flow

Source: [Flow of Theft Detection.pdf](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/Flow%20of%20Theft%20Detection.pdf)

## Purpose

This file is a working project brief distilled from the PDF so future coding work can reuse the intended business flow without needing to reread the document each time.

## End-to-End Flow

1. Someone opens the door.
2. Aqara is triggered and sends the trigger to `n8n`.
3. `n8n` searches the entry method in the database and saves the trigger info into the `Trigger` table.
   - If the entry is whitelisted, there is no need to run LLM input.
4. Wait 5 minutes.
5. Retrieve video from the Da Hua CCTV.
6. Check whether the video is ready.
   - If not ready, wait 5 more minutes and retry up to 3 times.
   - If still not ready after 3 retries, update `Trigger` status to `Issue`.
7. Save the video.
8. Run the script on the video.

## Entry Script Responsibilities

### Detect Enter

1. Create a session.
   - Update video link URL.
   - Get the start time.
2. Create a customer record.
3. Save ReID image.
4. Create the embedding value in database table `Customer_Gallery`.

### Detect Exit

1. Find the session and customer.
2. Save ReID image.
3. Update customer.
4. Check whether all customers have left.
   - If all have left:
     - Update session.
     - Update closed time.
     - Update video link URL for close.

## Kiosk Script Responsibilities

This starts after the session is detected as closed.

1. Retrieve kiosk video between `Start Time` and `Closed Time`.
2. Run kiosk script:
   - Get the transactions that happen between `Start Time` and `Closed Time`.
   - Detect the related customer using embeddings.
   - Calculate total item brought out from the store.
     - Update that value into `Session`.
   - Compare kiosk total item vs transaction total item.

### Session Result Logic

1. If kiosk total matches transactions:
   - Update session as `Not Detected`.
2. If kiosk quantity is more than transactions:
   - Update session as `Detected`.
3. If transactions quantity is more than kiosk:
   - Update session as `Need Review`.

## Error Handling Rule

If one component errors, stop the rest, unless manually reset to the session that should continue.

## Database Design

### `Whitelist_Entry`

- `ID`
- `Method`
- `Entry_ID`
- `Status`
- `Created_at`
- `Updated_at`

### `CCTV`

- `ID`
- `LocationId`
- `Delayed Time`
- `Section`

### `Trigger`

- `ID`
- `LocationId`
- `Created_Time`
- `Status`
  - `Pending`
  - `Done`
  - `Issue`

### `Session`

- `Id`
- `Entry Video Link`
- `Trigger_Id`
- `Start_time`
- `End_time`
- `Total_item_brought`
- `Actual_items_brought`
- `Status`
  - `Detected`
  - `Not Detected`
  - `Issue`
  - `Whitelisted`
- `Total_customer`
- `Created_at`

### `Session_Transaction`

- `Id`
- `Session_id`
- `ReceiptNumber`
- `Created_at`

### `Exit_Video`

- `Id`
- `Url`
- `Session_Id`
- `created_at`

### `Customer_Gallery`

- `Id`
- `Session_id`
- `Embedding_Osnet`
- `Embedding_Fashion`
- `Image_URL`
- `Created_at`

## Current System Intent

The intended architecture is:

1. Aqara trigger creates a `Trigger`.
2. Entry/exit video processing creates and closes a `Session`.
3. ReID images and embeddings are stored in `Customer_Gallery`.
4. Kiosk analysis estimates what the group brought out.
5. Kiosk output is compared against real transaction data.
6. Final session status is set based on mismatch severity.

## Phase 2 Items

### Whitelist More People

Possible extra whitelists:

- Supplier
- Own staff
- Worker
- Cleaner

### Sneak-In Detection

- If someone sneaks in, consider them part of the group.
- Open issue: how to know this reliably.

### Door Held Open for Others

- If someone opens the door for a friend or another person, consider them part of the group.
- Open issue: how to detect this reliably.

### Concurrent Sessions

Example:

- `P1` enters and loiters.
- `P2` enters, grabs items, buys, and leaves.
- `P1` only comes and pays later.

Idea:

- Consider concurrency logic to save time and energy.
- Open issue: how to model this properly.

### Unknown Door Trigger Scenario

Case:

- Door opens or Aqara switch is toggled without a normal customer flow.

Idea:

- Consider assigning this as `Unknown`.

### Product Identification

- Train Grounding DINO using product images.

## Practical Notes for Future Implementation

- The session is the main unit tying together trigger, entry, kiosk, exit, and transactions.
- ReID embeddings are expected to connect entry/exit identities to kiosk identities.
- Whitelist logic should short-circuit expensive LLM analysis where possible.
- Kiosk analysis is not the final truth by itself; it is meant to be reconciled with transaction records.
- Failure handling is strict by design: stop downstream steps if an upstream dependency fails.

