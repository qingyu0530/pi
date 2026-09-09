use pi_ai::{
    AssistantImages, ImageContent, ImagesContext, ImagesStopReason, TextContent, Usage, UsageCost,
    UserContent,
};
use serde_json::json;

#[test]
fn images_context_uses_user_content_blocks() {
    let context = ImagesContext {
        input: vec![UserContent::Text(TextContent {
            text: "生成一张猫的图片".to_owned(),
            text_signature: None,
        })],
    };

    let expected = json!({
        "input": [ { "type": "text", "text": "生成一张猫的图片" } ]
    });

    assert_eq!(serde_json::to_value(&context).unwrap(), expected);
}

#[test]
fn assistant_images_matches_the_typescript_json_shape() {
    let image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a9WQAAAAASUVORK5CYII=";
    let result = AssistantImages {
        api: "openrouter-images".to_owned(),
        provider: "openrouter".to_owned(),
        model: "gpt-image-1".to_owned(),
        output: vec![UserContent::Image(ImageContent {
            data: image_data.to_owned(),
            mime_type: "image/png".to_owned(),
        })],
        response_id: None,
        usage: Some(Usage {
            input: 10,
            output: 0,
            cache_read: 0,
            cache_write: 0,
            cache_write_1h: None,
            reasoning: None,
            total_tokens: 10,
            cost: UsageCost {
                input: 0.001,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.001,
            },
        }),
        stop_reason: ImagesStopReason::Stop,
        error_message: None,
        timestamp: 1_700_000_000_000,
    };

    let expected = json!({
        "api": "openrouter-images",
        "provider": "openrouter",
        "model": "gpt-image-1",
        "output": [ { "type": "image", "data": image_data, "mimeType": "image/png" } ],
        "usage": {
            "input": 10, "output": 0, "cacheRead": 0, "cacheWrite": 0,
            "totalTokens": 10,
            "cost": { "input": 0.001, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.001 }
        },
        "stopReason": "stop",
        "timestamp": 1700000000000_u64
    });

    assert_eq!(serde_json::to_value(&result).unwrap(), expected);
    assert_eq!(
        serde_json::from_value::<AssistantImages>(expected).unwrap(),
        result
    );
}
