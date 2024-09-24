import logging

def evaluate_matches(avg_match_score, max_match_score, avg_match_response, max_match_response, avg_match_count, max_match_count, avg_match_flag, max_match_flag, threshold):
    if avg_match_score >= threshold and max_match_score < avg_match_score:
        logging.info(
            f"Avg Match Score: {avg_match_score:.4f}, Avg Match Response: '{avg_match_response}"
            f"Match Count: {avg_match_count}, Avg Match: {avg_match_flag}"
        )
        print("This is from Average Match Response")
        return avg_match_response

    elif max_match_score > avg_match_score:
        logging.info(
            f"Max Match Score: {max_match_score:.4f}, Max Match Response: '{max_match_response}'" 
            f"Match Count: {max_match_count}, Max Match: {max_match_flag}"
        )
        print("This is from Max Match Response")
        return max_match_response

    else:
        logging.warning(
            f"No suitable match found for query with score above threshold: {threshold}"
        )
        return None
