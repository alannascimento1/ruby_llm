# frozen_string_literal: true

module RubyLLM
  module Providers
    class Mistral
      # Dedicated support for the Mistral OCR models.
      module OCR
        TEXTUAL_KEYS = %w[text markdown content value raw_text rawText string plain_text plainText].freeze
        COLLECTION_KEYS = %w[blocks pages lines paragraphs items elements entries sections text_blocks textBlocks].freeze
        SKIP_RECURSION_KEYS = %w[image_url image data file_data fileData url source bytes raw binary].freeze

        private

        def ocr_model?(model_id)
          model_id.match?(/mistral-ocr/)
        end

        def complete_ocr(messages, model:, params:, headers:, streaming: false)
          ensure_non_streaming_ocr!(streaming)

          content = latest_user_content(messages)
          attachment = content&.attachments&.first

          raise RubyLLM::Error.new(nil, 'Mistral OCR models require an attachment.') unless attachment

          payload = build_ocr_payload(attachment, model:, params:)
          response = connection.post('ocr', payload) do |request|
            request.headers.merge!(headers) if headers&.any?
          end

          build_ocr_message(response, model:)
        end

        def ensure_non_streaming_ocr!(streaming)
          return unless streaming

          raise RubyLLM::Error.new(nil, 'Mistral OCR models do not support streaming responses.')
        end

        def latest_user_content(messages)
          message = messages.reverse.find { |msg| msg.role == :user }
          return unless message

          raw_content = message.instance_variable_get(:@content)
          return raw_content if raw_content.is_a?(RubyLLM::Content)

          value = message.content
          return value if value.is_a?(RubyLLM::Content)
          return RubyLLM::Content.new(value) if value.is_a?(String) && !value.empty?

          nil
        end

        def build_ocr_payload(attachment, model:, params:)
          base_payload = {
            model: model.id,
            document: document_payload_for(attachment)
          }

          RubyLLM::Utils.deep_merge(base_payload, params || {})
        end

        def document_payload_for(attachment)
          case attachment.type
          when :image
            {
              type: 'image_url',
              image_url: attachment.url? ? attachment.source.to_s : attachment.for_llm
            }
          when :pdf
            {
              type: 'file',
              file: {
                filename: attachment.filename,
                file_data: attachment.for_llm
              }
            }
          else
            raise RubyLLM::UnsupportedAttachmentError, "Attachment type #{attachment.type} is not supported by Mistral OCR"
          end
        end

        def build_ocr_message(response, model:)
          body = response.body
          text = extract_ocr_text(body)

          RubyLLM::Message.new(
            role: :assistant,
            content: text || JSON.pretty_generate(body),
            input_tokens: body.dig('usage', 'prompt_tokens') || body.dig('usage', 'input_tokens'),
            output_tokens: body.dig('usage', 'completion_tokens') || body.dig('usage', 'output_tokens'),
            model_id: model.id,
            raw: response
          )
        end

        def extract_ocr_text(payload)
          fragments = gather_text_fragments(payload)
          unique = fragments.map(&:strip).reject(&:empty?).uniq
          return if unique.empty?

          unique.join("\n\n")
        end

        def gather_text_fragments(node)
          case node
          when Hash
            fragments_from_hash(node)
          when Array
            node.flat_map { |item| gather_text_fragments(item) }
          when String
            [node]
          else
            []
          end
        end

        def fragments_from_hash(hash)
          fragments = []

          TEXTUAL_KEYS.each do |key|
            value = hash[key]
            fragments.concat(gather_text_fragments(value)) if value
          end

          COLLECTION_KEYS.each do |key|
            value = hash[key]
            fragments.concat(gather_text_fragments(value)) if value
          end

          hash.each do |key, value|
            next if TEXTUAL_KEYS.include?(key)
            next if COLLECTION_KEYS.include?(key)
            next if SKIP_RECURSION_KEYS.include?(key)

            fragments.concat(gather_text_fragments(value)) if value.is_a?(Hash) || value.is_a?(Array)
          end

          fragments
        end
      end
    end
  end
end
